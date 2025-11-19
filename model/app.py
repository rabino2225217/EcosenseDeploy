from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2, numpy as np, base64, os, tempfile, json
from flask_cors import CORS
from pathlib import Path
import rasterio
from rasterio.transform import xy
from rasterio import features
from pyproj import Transformer
import torch
from torchvision.ops import nms
import geopandas as gpd
from ensemble_boxes import weighted_boxes_fusion
from dotenv import load_dotenv

#Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

#Model paths
MODEL_PATHS = {
    "tree": os.environ["MODEL_TREE_PATH"],
    "crop": os.environ["MODEL_CROP_PATH"],
}

#Load models
MODELS = {}
for key, path in MODEL_PATHS.items():
    if Path(path).exists():
        MODELS[key] = YOLO(path)
        print(f"Loaded model '{key}'")
    else:
        print(f"Model path not found for '{key}': {path}")

#BBox color
COLOR_MAP = {
    "trees-5pma": (0, 255, 255),   # #008000
    "Tree": (0, 255, 255),         # #00FFFF
    "Bokchoy": (165, 191, 0),      # #00BFA5
    "Potato": (0, 165, 255),       # #FFA500 
    "Lettuce": (0, 252, 124),      # #7CFC00
    "Romaine": (10, 47, 10),       # #0A2F0A
    "default": (255, 255, 255)     # #FFFFFFFF 
}

#YOLO prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    model_key = request.form.get("model", "tree")
    conf_threshold = float(request.form.get("confidence"))
    iou_threshold = float(request.form.get("iou"))

    if model_key not in MODEL_PATHS:
        return jsonify({"error": f"Invalid model '{model_key}'"}), 400
    if model_key not in MODELS:
        return jsonify({"error": f"Model '{model_key}' not loaded."}), 500

    model = MODELS[model_key]
    temp_path = None

    try:
        #Save Temporary File
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            temp_path = tmp.name
            file.save(temp_path)

        #Load GeoTIFF
        with rasterio.open(temp_path) as src:
            bands = src.count

            if bands >= 3:
                img = src.read([1, 2, 3])
            else:
                img = src.read()

            img = np.transpose(img, (1, 2, 0))
            transform = src.transform
            crs = src.crs

            try:
                nodata_mask = src.read_masks(1)
            except Exception:
                nodata_mask = None

        #Prepare Image
        img = img.astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        H, W = img.shape[:2]

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        #Tiling
        tile_size = 1024
        overlap_percent = 0.3
        NMS_IOU_THRESHOLD = 0.5

        def get_tiles(H, W, tile_size=1024, overlap_percent=0.3):
            tiles = []

            #Convert percentage overlap to pixel overlap
            overlap_px = int(tile_size * overlap_percent)

            step = tile_size - overlap_px

            for y in range(0, H, step):
                for x in range(0, W, step):
                    tiles.append((
                        x, y,
                        min(x + tile_size, W),
                        min(y + tile_size, H),
                    ))
            return tiles

        #Run predictions on tiles
        all_dets = []

        for x1, y1, x2, y2 in get_tiles(H, W):
            tile = img[y1:y2, x1:x2]

            #Skip empty nodata tiles
            if nodata_mask is not None:
                mask_tile = nodata_mask[y1:y2, x1:x2]
                if np.mean(mask_tile == 0) > 0.6:
                    continue

            results = model.predict(
                source=tile,
                conf=conf_threshold,
                iou=iou_threshold,
                save=False
            )

            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            #Convert tile coords to full image coords
            for (bx1, by1, bx2, by2), cls, cf in zip(boxes, classes, confs):

                bx1 = max(0, min(bx1, x2 - x1))
                by1 = max(0, min(by1, y2 - y1))
                bx2 = max(0, min(bx2, x2 - x1))
                by2 = max(0, min(by2, y2 - y1))

                X1 = int(bx1 + x1)
                Y1 = int(by1 + y1)
                X2 = int(bx2 + x1)
                Y2 = int(by2 + y1)

                if X2 <= X1 or Y2 <= Y1:
                    continue

                if nodata_mask is not None:
                    crop = nodata_mask[Y1:Y2, X1:X2]
                    if np.mean(crop == 0) > 0.5:
                        continue

                all_dets.append({
                    "x1": X1, "y1": Y1,
                    "x2": X2, "y2": Y2,
                    "cls": int(cls),
                    "confidence": float(cf)
                })

        #Global merging of detections
        merged = []
        final = []

        if len(all_dets) > 0:

            # WBF input
            boxes_list = [[
                [d["x1"]/W, d["y1"]/H, d["x2"]/W, d["y2"]/H]
                for d in all_dets
            ]]
            scores_list = [[d["confidence"] for d in all_dets]]
            labels_list = [[d["cls"] for d in all_dets]]

            #Weighted Boxes Fusion (WBF)
            wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                iou_thr=0.45,      
                skip_box_thr=0.05
            )

            #Convert WBF boxes to original scale
            temp_dets = []
            for box, score, label in zip(wbf_boxes, wbf_scores, wbf_labels):
                x1 = int(box[0] * W)
                y1 = int(box[1] * H)
                x2 = int(box[2] * W)
                y2 = int(box[3] * H)
                temp_dets.append([x1, y1, x2, y2, float(score), int(label)])

            #Final non-maximum suppression (NMS)
            if len(temp_dets) > 0:
                arr = np.array(temp_dets)
                coords = torch.tensor(arr[:, :4], dtype=torch.float32)
                scores = torch.tensor(arr[:, 4], dtype=torch.float32)

                keep = nms(coords, scores, iou_threshold=0.40).cpu().numpy()

                max_w = W * 0.15
                max_h = H * 0.15

                #Filter and prepare final merged detections
                for idx in keep:
                    x1, y1, x2, y2, score, cls_id = temp_dets[idx]
                    if (x2 - x1) > max_w or (y2 - y1) > max_h:
                        continue

                    merged.append({
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "cls": int(cls_id),
                        "confidence": float(score)
                    })

        #GPS conversion and final output
        for det in merged:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            lon_proj, lat_proj = xy(transform, cy, cx)
            lon, lat = transformer.transform(lon_proj, lat_proj)

            final.append({
                "label": model.names[det["cls"]],
                "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "gps_coordinates": {"lat": lat, "lon": lon},
                "confidence": det["confidence"]
            })

        #Draw bounding boxes on image
        for det in merged:
            color = COLOR_MAP.get(model.names[det["cls"]], COLOR_MAP["default"])
            cv2.rectangle(img, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 3)


        #Encode result image
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "detections": final,
            "image_size": {"width": W, "height": H},
            "metadata": {"crs": str(crs), "converted_to": "EPSG:4326"},
            "result_image": f"data:image/jpeg;base64,{img_base64}"
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        #Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


#Max file size of land cover upload
MAX_FILE_SIZE = int(os.environ["MAX_FILE_SIZE_MB"]) * 1024 * 1024

#Land cover conversion endpoint
@app.route("/convert", methods=["POST"])
def convert():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()
    land_type = request.form.get("type", "trees").lower()

    if not filename.endswith((".tif", ".tiff")):
        return jsonify({"error": "Only GeoTIFF (.tif or .tiff) files are accepted"}), 400

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_FILE_SIZE:
        return jsonify({"error": f"File size exceeds {MAX_FILE_SIZE // (1024*1024)} MB limit."}), 400

    #Map land type to raster value
    target_value = 2 if land_type == "trees" else 5 if land_type == "crops" else None
    if target_value is None:
        return jsonify({"error": "Invalid type. Must be 'trees' or 'crops'"}), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            temp_path = tmp.name
            file.save(temp_path)

        with rasterio.open(temp_path) as src:
            if src.count == 0:
                return jsonify({"error": "GeoTIFF has no bands."}), 400
            if src.nodata is None:
                return jsonify({"error": "GeoTIFF has no nodata value."}), 400

            #Read raster and create mask
            image = src.read(1)
            mask = image != src.nodata

            results = (
                {"properties": {"value": v}, "geometry": s}
                for s, v in features.shapes(image, mask=mask, transform=src.transform)
            )
            gdf = gpd.GeoDataFrame.from_features(results, crs=src.crs)
            gdf = gdf[gdf["value"] == target_value]

            if gdf.empty:
                return jsonify({
                    "error": f"No {land_type} detected in the provided GeoTIFF)."
                }), 400
    
            gdf = gdf.to_crs(epsg=4326)

            gdf["geometry"] = gdf.simplify(tolerance=0.0001, preserve_topology=True)

            #Prepare GeoJSON output
            geojson_output = {
                "type": "FeatureCollection",
                "name": land_type,
                "crs": {
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
                },
                "features": json.loads(gdf.to_json())["features"]
            }

        return jsonify(geojson_output)

    except Exception as e:
        print("Conversion error:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        #Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            

# #Flask host, port, and debug
# FLASK_HOST = os.environ["FLASK_HOST"]
# FLASK_PORT = int(os.environ["FLASK_PORT"])
# FLASK_DEBUG = os.environ["FLASK_DEBUG"].lower() in ["true", "1", "yes"] 

# #DEV SERVER
# if __name__ == "__main__":
#     app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
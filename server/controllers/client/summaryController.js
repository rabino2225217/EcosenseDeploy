const { Parser } = require("json2csv");
const mongoose = require("mongoose");
const Detection = require("../../models/detection-model");
const GeoJSON = require("../../models/geojson-model");
const Summary = require("../../models/summary-model");
const turf = require("@turf/turf");

//Save or update project summary
exports.saveSummary = async (req, res) => {
  try {
    const {
      project_id: projectIdRaw,
      land_covers = [],
      filters = [],
    } = req.body;

    if (!projectIdRaw) {
      return res.status(400).json({ error: "project_id is required" });
    }

    const projectId = new mongoose.Types.ObjectId(projectIdRaw);

    const formattedLandCovers = land_covers.map((lc) => ({
      name: lc.name || "Not Specified",
      geojson_id: lc.geojson_id || null,
      counts: lc.counts || {},
    }));

    const summary = await Summary.findOneAndUpdate(
      { project_id: projectId },
      {
        project_id: projectId,
        land_covers: formattedLandCovers,
        filters,
        recorded_at: new Date(),
      },
      { new: true, upsert: true, overwrite: true }
    );

    res.json({ success: true, summary });
  } catch (err) {
    console.error("Failed to save summary:", err);
    res.status(500).json({ error: "Failed to save summary" });
  }
};

//Get summary for a specific project
exports.getSummary = async (req, res) => {
  try {
    const { project_id: projectIdRaw } = req.query;
    if (!projectIdRaw) {
      return res.status(400).json({ error: "project_id is required" });
    }

    const projectId = new mongoose.Types.ObjectId(projectIdRaw);

    const summary = await Summary.findOne({ project_id: projectId }).populate(
      "project_id",
      "name description"
    );

    if (!summary) {
      return res.status(404).json({ error: "Summary not found" });
    }

    res.json(summary);
  } catch (err) {
    console.error("Failed to get summary:", err);
    res.status(500).json({ error: "Failed to get summary" });
  }
};

//Export detailed detections
exports.exportDetailedDetections = async (req, res) => {
  try {
    const { project_id: projectIdRaw } = req.params;
    const { name } = req.query;

    if (!projectIdRaw) {
      return res.status(400).json({ error: "project_id is required" });
    }

    const projectId = new mongoose.Types.ObjectId(projectIdRaw);

    const safeName = (name || projectIdRaw)
      .replace(/[^\w\s-]/g, "")
      .replace(/\s+/g, "_");

    const summary = await Summary.findOne({ project_id: projectId }).sort({
      recorded_at: -1,
    });
    const landCoverNames = summary?.land_covers?.map((lc) => lc.name) || [];

    const detections = await Detection.find({ project_id: projectId });
    if (!detections.length) {
      return res
        .status(404)
        .json({ error: "No detections found for this project" });
    }

    const buildRows = (records, landCover = "Not Specified") =>
      records.map((d) => ({
        "Land Cover": landCover,
        Species: d.label || "N/A",
        Longitude: d.gps_coordinates?.lon || "",
        Latitude: d.gps_coordinates?.lat || "",
        Date: d.date ? new Date(d.date).toLocaleDateString() : "N/A",
      }));

    let rows;

    if (!landCoverNames.length) {
      rows = buildRows(detections);
    } else {
      const geojsons = await GeoJSON.find({ name: { $in: landCoverNames } });
      if (!geojsons.length) {
        rows = buildRows(detections);
      } else {
        const polygons = geojsons.flatMap((g) =>
          (g.features || []).map((f) => ({
            ...f,
            _meta: { name: g.name },
          }))
        );

        const filtered = detections
          .map((d) => {
            if (!d.gps_coordinates) return null;
            const point = turf.point([
              d.gps_coordinates.lon,
              d.gps_coordinates.lat,
            ]);
            const matched = polygons.find((feature) => {
              const geom = feature.geometry;
              if (!geom) return false;
              if (geom.type === "Polygon") {
                return turf.booleanPointInPolygon(point, geom);
              } else if (geom.type === "MultiPolygon") {
                return turf.booleanPointInPolygon(
                  point,
                  turf.multiPolygon(geom.coordinates)
                );
              }
              return false;
            });
            if (!matched) return null;
            return {
              ...d.toObject(),
              land_cover: matched._meta?.name || "Unknown",
            };
          })
          .filter(Boolean);

        if (filtered.length) {
          rows = filtered.map((d) => ({
            "Land Cover": d.land_cover || "Not Specified",
            Species: d.label || "N/A",
            Longitude: d.gps_coordinates?.lon || "",
            Latitude: d.gps_coordinates?.lat || "",
            Date: d.date ? new Date(d.date).toLocaleDateString() : "N/A",
          }));
        } else {
          rows = [
            {
              "Land Cover": "N/A",
              Species: "N/A",
              Longitude: "",
              Latitude: "",
              Date: "N/A",
            },
          ];
        }
      }
    }

    const parser = new Parser({
      fields: ["Land Cover", "Species", "Longitude", "Latitude", "Date"],
    });

    const csv = parser.parse(rows);
    res.header("Content-Type", "text/csv");
    res.attachment(`${safeName}_Summary.csv`);
    res.send(csv);
  } catch (error) {
    console.error("Error exporting detections:", error);
    res.status(500).json({ error: "Failed to export detections" });
  }
};

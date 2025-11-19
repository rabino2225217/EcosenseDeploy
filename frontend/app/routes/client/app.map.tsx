import socket from "../../services/socket";
import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { useParams } from "react-router-dom";
import { Funnel, Loader, Map as MapIcon } from "lucide-react";
import { Checkbox } from "../../components/ui/checkbox";
import debounce from "lodash/debounce";
import "ol/ol.css";

import RBush from "rbush";
import Map from "ol/Map";
import View from "ol/View";
import TileLayer from "ol/layer/Tile";
import TileWMS from "ol/source/TileWMS";
import OSM from "ol/source/OSM";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import VectorImageLayer from "ol/layer/VectorImage";
import GeoJSON from "ol/format/GeoJSON";
import { defaults as defaultControls, Zoom } from "ol/control";
import { fromLonLat, transformExtent } from "ol/proj";
import { Style, Fill, Stroke, Circle as CircleStyle } from "ol/style";
import { Polygon, MultiPolygon, Geometry, Point } from "ol/geom";
import Feature from "ol/Feature";
import type { FeatureLike } from "ol/Feature";

const API_URL = import.meta.env.VITE_API_URL;
const DEFAULT_CENTER: [number, number] = [120.6, 16.42];
const DEFAULT_ZOOM = 11;
const CAR_EXTENT: [number, number, number, number] = [
  120.25, 16.18, 121.1, 18.72,
];
const DEFAULT_LANDCOVER = "Not Specified";

export default function AppMap() {
  const { id: projectId } = useParams<{ id: string }>();
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapInstance = useRef<Map | null>(null);
  const detectionsLayerRef = useRef<VectorLayer<VectorSource> | null>(null);
  const maskLayerRef = useRef<VectorLayer<VectorSource> | null>(null);
  const landCoverLayersRef = useRef<VectorLayer<VectorSource>[]>([]);
  const activeLayerRef = useRef<TileLayer<TileWMS> | null>(null);
  const parsedGeoJSONCache = useRef<Record<string, Feature<Geometry>[]>>({});
  const styleCache = useRef<Record<string, Style>>({});
  const detectionIndexRef = useRef<RBush<{
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
    feature: Feature<Geometry>;
  }> | null>(null);

  const [activeWms, setActiveWms] = useState(null);
  const [layers, setLayers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [visibleLayers, setVisibleLayers] = useState<Record<string, boolean>>(
    {}
  );
  const [layerOpen, setLayerOpen] = useState(false);
  const [filterRaw, setFilterRaw] = useState<{ selectedClasses: string[] }>({
    selectedClasses: [],
  });
  const [filter, setFilter] = useState<{ selectedClasses: string[] }>({
    selectedClasses: [],
  });
  const [classes, setClasses] = useState<string[]>([]);
  const [filterOpen, setFilterOpen] = useState(false);
  const [legendCounts, setLegendCounts] = useState<Record<string, number>>({});
  const [landCoverBreakdown, setLandCoverBreakdown] = useState<
    Record<string, Record<string, number>>
  >({});
  const [expandedLegend, setExpandedLegend] = useState(false);
  const [initStatus, setInitStatus] = useState({
    baseMap: false,
    landCovers: false,
    activeLayer: false,
    detections: false,
  });
  const [detectionsLoaded, setDetectionsLoaded] = useState(false);
  const [confidenceThresholdRaw, setConfidenceThresholdRaw] = useState(0.25);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const [mapReady, setMapReady] = useState(false);
  const [legendReady, setLegendReady] = useState(false);
  const [legendLoading, setLegendLoading] = useState(false);

  //Initialize map
  useEffect(() => {
    if (!mapRef.current) return;

    const carExtent3857 = transformExtent(CAR_EXTENT, "EPSG:4326", "EPSG:3857");

    const map = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({
          source: new OSM({
            wrapX: false,
            attributions: ["© OpenStreetMap contributors | Copernicus Data"],
          }),
        }),
      ],
      view: new View({
        extent: carExtent3857,
        center: fromLonLat(DEFAULT_CENTER),
        zoom: DEFAULT_ZOOM,
      }),
      controls: defaultControls({ zoom: false }).extend([
        new Zoom({ className: "ol-custom-zoom" }),
      ]),
    });

    mapInstance.current = map;
    map.once("rendercomplete", () => {
      setInitStatus((s) => ({ ...s, baseMap: true }));
    });

    //WMS source creation
    const createWmsSource = (layerName: string) =>
      new TileWMS({
        url: `${API_URL}/admin/wms/tiles`,
        params: {
          LAYERS: layerName,
          FORMAT: "image/png",
          TRANSPARENT: "true",
          VERSION: "1.3.0",
        },
        serverType: "geoserver",
        wrapX: false,
      });

    //Load currently active WMS layer
    fetch(`${API_URL}/admin/wms/active-layer`)
      .then((res) => res.json())
      .then((data) => {
        if (!data?.layerName) {
          console.warn("No active layer found");
          setInitStatus((s) => ({ ...s, activeLayer: true }));
          return;
        }

        const wmsLayer = new TileLayer({
          source: createWmsSource(data.layerName),
        });

        wmsLayer.setZIndex(0);
        mapInstance.current?.addLayer(wmsLayer);
        activeLayerRef.current = wmsLayer;
        setActiveWms(data.layerName);
        setInitStatus((s) => ({ ...s, activeLayer: true }));
      })
      .catch((err) => console.error("Failed to load active layer:", err));

    //SOCKET CONNECTION for WMS active layer updates
    if (socket) {
      socket.on("active-layer-updated", (data) => {
        console.log("Active layer updated:", data);

        if (!mapInstance.current || !data?.layerName) return;

        if (activeLayerRef.current) {
          mapInstance.current.removeLayer(activeLayerRef.current);
          activeLayerRef.current = null;
        }

        const newLayer = new TileLayer({
          source: createWmsSource(data.layerName),
        });
        newLayer.setZIndex(0);
        mapInstance.current.addLayer(newLayer);
        activeLayerRef.current = newLayer;
        setActiveWms(data.layerName);
      });
    }

    return () => {
      if (socket) socket.off("active-layer-updated");
      mapInstance.current?.setTarget(undefined);
    };
  }, []);

  //Fetch landcover layers
  useEffect(() => {
    setLoading(true);
    fetch(`${API_URL}/layer/get`)
      .then((res) => res.json())
      .then((data) => setLayers(data))
      .catch((err) => console.error("Failed to load land cover files:", err))
      .finally(() => {
        setInitStatus((s) => ({ ...s, landCovers: true }));
        setLoading(false);
      });
  }, []);

  //Create mask layer
  const createMaskLayer = useCallback((geometries: Geometry[]) => {
    const worldOuterRing = [
      [-20037508.34, -20037508.34],
      [20037508.34, -20037508.34],
      [20037508.34, 20037508.34],
      [-20037508.34, 20037508.34],
      [-20037508.34, -20037508.34],
    ];

    const holes: number[][][] = [];

    geometries.forEach((geom) => {
      if (geom instanceof Polygon) {
        const rings = geom.getCoordinates();
        if (Array.isArray(rings) && rings.length > 0) {
          rings.forEach((ring) => holes.push(ring));
        }
      } else if (geom instanceof MultiPolygon) {
        geom.getPolygons().forEach((poly) => {
          const rings = poly.getCoordinates();
          if (Array.isArray(rings) && rings.length > 0) {
            holes.push(rings[0]);
          }
        });
      }
    });

    const maskPolygon = new Polygon([worldOuterRing, ...holes]);

    const maskFeature = new Feature(maskPolygon);
    const maskSource = new VectorSource({ features: [maskFeature] });

    const maskLayer = new VectorLayer({
      source: maskSource,
      style: new Style({
        fill: new Fill({ color: "rgba(0, 0, 0, 0.6)" }),
        stroke: new Stroke({ color: "rgba(0, 0, 0, 0.8)", width: 0 }),
      }),
    });

    maskLayer.set("name", "mask");
    return maskLayer;
  }, []);

  //Update legend counts and save summary
  const updateLegendCounts = useCallback(async () => {
    const detectionsLayer = detectionsLayerRef.current;
    if (!detectionsLayer) {
      setLegendCounts({});
      setLegendReady(true);
      return;
    }

    const detectionFeatures = detectionsLayer.getSource()?.getFeatures() || [];
    if (detectionFeatures.length === 0) {
      setLegendCounts({});
      setLegendReady(true);
      return;
    }

    const selected = filter.selectedClasses;
    if (selected.length === 0) {
      setLegendCounts({});
      await fetch(`${API_URL}/summary/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: projectId,
          land_covers: [],
          filters: [],
        }),
      }).catch((err) => console.error("Failed to save empty summary:", err));
      return;
    }

    const detectionIndex = detectionIndexRef.current;
    if (!detectionIndex) {
      console.warn("Detection index not built yet");
      return;
    }

    const activeLandCovers: {
      name: string;
      geojson_id?: string | null;
      geometries: Geometry[];
    }[] = [];

    landCoverLayersRef.current.forEach((lcLayer) => {
      const isOnMap = !!mapInstance.current
        ?.getLayers()
        .getArray()
        .includes(lcLayer);
      if (!isOnMap) return;

      const feats = lcLayer.getSource()?.getFeatures() || [];
      const geometries = feats
        .map((f) => f.getGeometry())
        .filter((g): g is Geometry => !!g);

      activeLandCovers.push({
        name: lcLayer.get("name"),
        geojson_id: lcLayer.get("id") || null,
        geometries,
      });
    });

    const landCoverCounts: Record<string, Record<string, number>> = {};
    activeLandCovers.forEach((lc) => (landCoverCounts[lc.name] = {}));

    activeLandCovers.forEach((lc) => {
      lc.geometries.forEach((geom) => {
        if (!(geom instanceof Polygon || geom instanceof MultiPolygon)) return;

        const extent = geom.getExtent();
        const candidates = detectionIndex.search({
          minX: extent[0],
          minY: extent[1],
          maxX: extent[2],
          maxY: extent[3],
        });

        candidates.forEach(({ feature }) => {
          const label = feature.get("label");
          if (!label || !selected.includes(label)) return;

          const fGeom = feature.getGeometry();
          if (
            fGeom instanceof Point &&
            geom.intersectsCoordinate(fGeom.getCoordinates())
          ) {
            landCoverCounts[lc.name][label] =
              (landCoverCounts[lc.name][label] || 0) + 1;
          }
        });
      });
    });

    if (activeLandCovers.length === 0) {
      const allDetections = detectionIndex.all().map((i) => i.feature);
      allDetections.forEach((f) => {
        const cls = f.get("label");
        if (!cls || !selected.includes(cls)) return;
        if (!landCoverCounts[DEFAULT_LANDCOVER])
          landCoverCounts[DEFAULT_LANDCOVER] = {};
        landCoverCounts[DEFAULT_LANDCOVER][cls] =
          (landCoverCounts[DEFAULT_LANDCOVER][cls] || 0) + 1;
      });
    }

    const totalCounts: Record<string, number> = {};
    Object.values(landCoverCounts).forEach((lcCounts) => {
      Object.entries(lcCounts).forEach(([cls, count]) => {
        totalCounts[cls] = (totalCounts[cls] || 0) + count;
      });
    });

    setLegendCounts(totalCounts);
    setLandCoverBreakdown(landCoverCounts);
    setLegendReady(true);

    const land_covers = Object.entries(landCoverCounts).map(
      ([name, counts]) => ({
        name,
        geojson_id:
          activeLandCovers.find((lc) => lc.name === name)?.geojson_id || null,
        counts,
      })
    );

    await fetch(`${API_URL}/summary/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: projectId,
        land_covers,
        filters: selected,
      }),
    }).catch((err) => console.error("Failed to save summary:", err));
  }, [projectId, filter.selectedClasses, landCoverLayersRef, mapInstance]);

  //Legend Debounce
  const updateLegendCountsRef = useRef(updateLegendCounts);
  useEffect(() => {
    updateLegendCountsRef.current = updateLegendCounts;
  }, [updateLegendCounts]);

  const debouncedUpdateLegendCounts = useRef(
    debounce(async () => {
      if (updateLegendCountsRef.current) {
        await updateLegendCountsRef.current();
      }
    }, 400)
  ).current;

  //SOCKET CONNECTION for landcover updates
  useEffect(() => {
    const handleUpdate = (data: any) => {
      if (!data || !data.action || !data.layer) return;

      switch (data.action) {
        case "add":
          setLayers((prev) => [...prev, data.layer]);
          setVisibleLayers((prev) => ({ ...prev, [data.layer.name]: false }));
          break;

        case "rename":
          setLayers((prev) =>
            prev.map((l) =>
              l._id === data.layer._id ? { ...l, name: data.layer.name } : l
            )
          );

          setVisibleLayers((prev) => {
            const updated = { ...prev };

            const oldKey = Object.keys(prev).find(
              (k) =>
                k.trim().toLowerCase() ===
                data.layer.oldName?.trim().toLowerCase()
            );

            if (oldKey) {
              updated[data.layer.name] = prev[oldKey];
              delete updated[oldKey];
            } else if (!(data.layer.name in updated)) {
              updated[data.layer.name] = updated[data.layer.name] ?? false;
            }

            return updated;
          });

          if (mapInstance.current) {
            const layerToRename = mapInstance.current
              .getLayers()
              .getArray()
              .find((l: any) => l.get("id") === data.layer._id);

            if (layerToRename) layerToRename.set("name", data.layer.name);
          }
          break;

        case "delete":
          setLayers((prev) => prev.filter((l) => l._id !== data.layer._id));
          setVisibleLayers((prev) => {
            const updated = { ...prev };
            delete updated[data.layer.name];
            return updated;
          });

          if (!mapInstance.current) break;

          const remainingGeometries: Geometry[] = [];
          landCoverLayersRef.current.forEach((l) => {
            if (l.get("id") !== data.layer._id) {
              const feats = l.getSource()?.getFeatures() ?? [];
              feats.forEach((f) => {
                const g = f.getGeometry();
                if (g) remainingGeometries.push(g);
              });
            }
          });

          const layerToRemove = landCoverLayersRef.current.find(
            (l) => l.get("id") === data.layer._id
          );
          if (layerToRemove) mapInstance.current.removeLayer(layerToRemove);
          landCoverLayersRef.current = landCoverLayersRef.current.filter(
            (l) => l.get("id") !== data.layer._id
          );

          if (maskLayerRef.current) {
            mapInstance.current.removeLayer(maskLayerRef.current);
            maskLayerRef.current = null;
          }

          if (remainingGeometries.length > 0) {
            const newMask = createMaskLayer(remainingGeometries);
            newMask.setZIndex(9999);
            maskLayerRef.current = newMask;
            mapInstance.current.addLayer(newMask);
          }
          debouncedUpdateLegendCounts();
          break;
      }
    };

    socket.on("landcover:update", handleUpdate);
    return () => {
      socket.off("landcover:update", handleUpdate);
    };
  }, [createMaskLayer, debouncedUpdateLegendCounts]);

  //Mask rebuild
  const updateMaskLayer = useCallback(
    (visible: Record<string, boolean>) => {
      if (!mapInstance.current) return;

      const activeGeometries: Geometry[] = [];
      landCoverLayersRef.current.forEach((layer) => {
        if (visible[layer.get("name")]) {
          const feats = layer.getSource()?.getFeatures() || [];
          feats.forEach((f) => {
            const g = f.getGeometry();
            if (g) activeGeometries.push(g);
          });
        }
      });

      if (maskLayerRef.current) {
        mapInstance.current.removeLayer(maskLayerRef.current);
        maskLayerRef.current = null;
      }

      if (activeGeometries.length > 0) {
        const newMask = createMaskLayer(activeGeometries);
        newMask.setZIndex(9999);
        maskLayerRef.current = newMask;
        mapInstance.current.addLayer(newMask);
      }
    },
    [createMaskLayer]
  );

  //Toggle landcover overlay
  const handleToggle = useCallback(
    async (layer: any) => {
      if (!mapInstance.current) return;

      setLegendLoading(true);
      setLegendCounts({});
      setLegendReady(false);

      const current = { ...visibleLayers };
      const isVisible = !current[layer.name];
      current[layer.name] = isVisible;
      setVisibleLayers(current);

      const existing = mapInstance.current
        .getLayers()
        .getArray()
        .find((l: any) => l.get("name") === layer.name);

      if (isVisible && !existing) {
        let features = parsedGeoJSONCache.current[layer.name];
        if (!features) {
          features = await new Promise<Feature<Geometry>[]>((resolve) => {
            setTimeout(() => {
              const parsed = new GeoJSON().readFeatures(layer, {
                dataProjection: "EPSG:4326",
                featureProjection: "EPSG:3857",
              });
              parsedGeoJSONCache.current[layer.name] = parsed;
              resolve(parsed);
            }, 0);
          });
        }

        const vectorSource = new VectorSource({ features });

        const vectorLayer = new VectorImageLayer({
          source: vectorSource,
          style: new Style({
            stroke: new Stroke({ color: "#32CD32", width: 3 }),
          }),
          imageRatio: 2,
        });

        vectorLayer.setZIndex(10);
        vectorLayer.set("name", layer.name);
        vectorLayer.set("id", layer._id);
        mapInstance.current.addLayer(vectorLayer);

        landCoverLayersRef.current.push(
          vectorLayer as unknown as VectorLayer<VectorSource>
        );

        const extent = vectorSource.getExtent();
        if (extent && extent.every((v) => isFinite(v))) {
          requestAnimationFrame(() => {
            mapInstance.current?.getView().fit(extent, {
              padding: [50, 50, 50, 50],
              duration: 600,
            });
          });
        }
      } else if (!isVisible && existing) {
        mapInstance.current.removeLayer(existing);
        landCoverLayersRef.current = landCoverLayersRef.current.filter(
          (l) => l.get("name") !== layer.name
        );
      }
      updateMaskLayer(current);

      Promise.resolve(debouncedUpdateLegendCounts()).finally(() =>
        setLegendLoading(false)
      );
    },
    [mapInstance, visibleLayers, debouncedUpdateLegendCounts, updateMaskLayer]
  );

  //Convert detections to point
  const detectionStyle = useCallback((feature: FeatureLike): Style => {
    const color = feature.get("color");

    if (!styleCache.current[color]) {
      styleCache.current[color] = new Style({
        image: new CircleStyle({
          radius: 3,
          fill: new Fill({ color }),
          stroke: new Stroke({ color: "rgba(255,255,255,0.6)", width: 0.3 }),
        }),
      });
    }
    return styleCache.current[color];
  }, []);

  //Confidence debounce
  useEffect(() => {
    const handle = setTimeout(() => {
      setConfidenceThreshold(confidenceThresholdRaw);
    }, 300);
    return () => clearTimeout(handle);
  }, [confidenceThresholdRaw]);

  //Class filter debounce
  useEffect(() => {
    const handle = setTimeout(() => {
      setFilter(filterRaw);
    }, 300);

    return () => clearTimeout(handle);
  }, [filterRaw]);

  //Fetch all detections
  useEffect(() => {
    if (!mapInstance.current || !projectId) return;

    if (filter.selectedClasses.length > 0) return;

    if (classes.length > 0 && filter.selectedClasses.length === 0) {
      if (detectionsLayerRef.current && mapInstance.current) {
        mapInstance.current.removeLayer(detectionsLayerRef.current);
        detectionsLayerRef.current = null;
      }
      setDetectionsLoaded(true);
      setLegendCounts({});
      return;
    }

    fetch(
      `${API_URL}/layer/detections?project_id=${projectId}&min_confidence=${confidenceThreshold}`
    )
      .then((res) => res.json())
      .then((geojson) => {
        setDetectionsLoaded(true);
        const vectorSource = new VectorSource({
          features: new GeoJSON().readFeatures(geojson, {
            featureProjection: "EPSG:3857",
          }),
        });

        const features = vectorSource.getFeatures();
        buildDetectionIndex(features);

        const detectionsLayer = new VectorImageLayer({
          source: vectorSource,
          style: detectionStyle,
          imageRatio: 3,
        });
        detectionsLayer.set("name", "detections");
        detectionsLayer.setZIndex(10000);

        if (detectionsLayerRef.current) {
          mapInstance.current?.removeLayer(detectionsLayerRef.current);
        }

        mapInstance.current?.addLayer(detectionsLayer);
        detectionsLayerRef.current = detectionsLayer as any;

        if (features.length === 0) {
          setInitStatus((s) => ({ ...s, detections: true }));
          setLegendCounts({});
          setLegendReady(true);
          return;
        }

        const waitForDetections = () => {
          const ready =
            (detectionIndexRef.current?.all().length || 0) > 0 &&
            (detectionsLayerRef.current?.getSource()?.getFeatures()?.length ||
              0) > 0;

          if (!ready) {
            setTimeout(waitForDetections, 200);
            return;
          }

          mapInstance.current?.once("rendercomplete", () => {
            debouncedUpdateLegendCounts();
            setInitStatus((s) => ({ ...s, detections: true }));
          });
        };
        waitForDetections();
      })
      .catch((err) => console.error("Failed to load detections:", err));
  }, [projectId, confidenceThreshold, activeWms, filter.selectedClasses]);

  //Build RBush index for detections
  const buildDetectionIndex = (detections: Feature<Geometry>[]) => {
    const items = detections
      .map((f) => {
        const extent = f.getGeometry()?.getExtent();
        if (!extent) return null;
        return {
          minX: extent[0],
          minY: extent[1],
          maxX: extent[2],
          maxY: extent[3],
          feature: f,
        };
      })
      .filter(Boolean) as {
      minX: number;
      minY: number;
      maxX: number;
      maxY: number;
      feature: Feature<Geometry>;
    }[];

    const tree = new RBush<(typeof items)[0]>();
    tree.load(items);
    detectionIndexRef.current = tree;
  };

  //Re-fetch detections when apply filter changes
  const lastFetchId = useRef(0);
  useEffect(() => {
    if (!mapInstance.current || !projectId) return;

    const run = async () => {
      if (classes.length > 0 && filter.selectedClasses.length === 0) {
        lastFetchId.current++;

        if (detectionsLayerRef.current && mapInstance.current) {
          mapInstance.current.removeLayer(detectionsLayerRef.current);
          detectionsLayerRef.current = null;
        }

        setLegendCounts({});

        //Save empty summary when no classes selected
        try {
          await fetch(`${API_URL}/summary/save`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              project_id: projectId,
              land_covers: [],
              filters: [],
            }),
          });
        } catch (err) {
          console.error("Failed to save empty summary:", err);
        }
        return;
      }

      if (classes.length === 0) return;
      const fetchId = ++lastFetchId.current;

      const url = new URL(`${API_URL}/layer/filtered-detections`);
      url.searchParams.append("project_id", projectId);
      url.searchParams.append("labels", filter.selectedClasses.join(","));
      url.searchParams.append("min_confidence", confidenceThreshold.toString());

      fetch(url.toString())
        .then((res) => res.json())
        .then((geojson) => {
          if (fetchId !== lastFetchId.current) return;
          if (!mapInstance.current) return;

          const vectorSource = new VectorSource({
            features: new GeoJSON().readFeatures(geojson, {
              featureProjection: "EPSG:3857",
            }),
          });

          const filteredLayer = new VectorImageLayer({
            source: vectorSource,
            style: detectionStyle,
            imageRatio: 3,
          });
          filteredLayer.set("name", "detections");
          filteredLayer.setZIndex(10000);

          if (detectionsLayerRef.current) {
            mapInstance.current.removeLayer(detectionsLayerRef.current);
          }

          mapInstance.current.addLayer(filteredLayer);
          detectionsLayerRef.current = filteredLayer as any;

          if (mapInstance.current) {
            mapInstance.current?.once("rendercomplete", () => {
              debouncedUpdateLegendCounts();
            });
          }
        })
        .catch((err) =>
          console.error("Failed to fetch filtered detections:", err)
        );
    };
    run();
  }, [
    filter.selectedClasses,
    projectId,
    classes,
    confidenceThreshold,
    activeWms,
  ]);

  useEffect(() => {
    if (Object.values(initStatus).every(Boolean)) {
      setTimeout(() => setMapReady(true), 100);
    }
  }, [initStatus]);

  //Fetch detection classes for filter options
  useEffect(() => {
    if (!projectId) return;

    fetch(`${API_URL}/layer/classes?project_id=${projectId}`)
      .then((res) => res.json())
      .then((data) => {
        let clsList: string[] = [];
        if (Array.isArray(data)) clsList = data;
        else if (Array.isArray(data.classes)) clsList = data.classes;

        setClasses(clsList);

        setFilterRaw({ selectedClasses: clsList });
        setFilter({ selectedClasses: clsList });
      })
      .catch((err) => console.error("Failed to load classes:", err));
  }, [projectId]);

  return (
    <div className="w-full h-full relative select-none">
      {/* Floating buttons */}
      <div className="absolute top-2 left-2 z-10 flex flex-col sm:flex-row sm:space-x-3 space-y-2 sm:space-y-0">
        {/* Land Cover Button */}
        <div className="relative w-[11rem] sm:w-[13rem]">
          <button
            onClick={() => setLayerOpen(!layerOpen)}
            className="flex items-center w-full bg-[#111]/90 hover:bg-[#1a1a1a]
                      text-white px-3 sm:px-4 py-1 sm:py-1.5 rounded-md shadow-md backdrop-blur-sm
                      border border-white/10 transform transition-transform duration-300
                      ease-in-out hover:scale-[1.02] active:scale-95 cursor-pointer"
          >
            <div className="flex items-center">
              <MapIcon className="w-3 h-3 sm:w-3.5 sm:h-3.5 mr-1.5 sm:mr-2" />
              <span
                className="text-[10px] sm:text-[12px] font-medium"
                data-qg="map-view"
              >
                Land Cover
              </span>
            </div>
            <svg
              className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ml-auto transition-transform duration-300 ease-in-out ${
                layerOpen ? "rotate-180" : "rotate-0"
              }`}
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>

          {/* Land Cover Dropdown */}
          {layerOpen && (
            <div
              className="mt-1 w-[11rem] sm:w-[13rem] bg-[#0a0a0a]/95 text-white rounded-md
                        shadow-lg p-2 sm:p-3 space-y-1.5 sm:space-y-2 border border-white/10
                        backdrop-blur-md z-20"
            >
              <h3 className="text-[10px] sm:text-[11px] uppercase tracking-wide text-gray-300 mb-1 sm:mb-2 pb-1 border-b border-gray-700">
                Available Layers
              </h3>

              {loading ? (
                <div className="flex items-center justify-center gap-2 py-3 text-[10px] sm:text-xs text-gray-400">
                  Loading land covers
                  <Loader className="w-4 h-4 animate-spin text-gray-400" />
                </div>
              ) : layers.length === 0 ? (
                <div className="text-center py-2 sm:py-3 text-[10px] sm:text-xs text-gray-400">
                  No land cover available
                </div>
              ) : (
                <>
                  {layers.map((layer) => {
                    const isChecked = visibleLayers[layer.name] || false;

                    return (
                      <div
                        key={layer._id}
                        className="flex items-center justify-between px-2 py-1 rounded-md hover:bg-white/10 transition text-[10px] sm:text-[11px]"
                      >
                        <div className="flex items-center gap-3 w-full">
                          <Checkbox
                            checked={isChecked}
                            onCheckedChange={() => handleToggle(layer)}
                            className="h-3 w-3 sm:h-3.5 sm:w-3.5 rounded-[3px]
                                      border-gray-400
                                      data-[state=checked]:bg-emerald-500
                                      data-[state=checked]:border-emerald-500
                                      [&>svg]:scale-50 [&>svg]:stroke-[2.5] [&>svg]:text-white cursor-pointer"
                          />
                          <span
                            className="truncate max-w-[8rem] sm:max-w-[10rem] text-ellipsis overflow-hidden"
                            title={layer.name}
                          >
                            {layer.name}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </>
              )}
            </div>
          )}
        </div>

        {/* Filter Button */}
        <div className="relative w-[11rem] sm:w-[13rem]">
          <button
            onClick={() => setFilterOpen(!filterOpen)}
            className="flex items-center w-full bg-[#111]/90 hover:bg-[#1a1a1a]
                      text-white px-3 sm:px-4 py-1 sm:py-1.5 rounded-md shadow-md backdrop-blur-sm
                      border border-white/10 transform transition-transform duration-300
                      ease-in-out hover:scale-[1.02] active:scale-95 cursor-pointer"
          >
            <div className="flex items-center">
              <Funnel className="w-3 h-3 sm:w-3.5 sm:h-3.5 mr-1.5 sm:mr-2" />
              <span
                className="text-[10px] sm:text-[12px] font-medium"
                data-qg="filters-panel"
              >
                Filter
              </span>
            </div>
            <svg
              className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ml-auto transition-transform duration-300 ease-in-out ${
                filterOpen ? "rotate-180" : "rotate-0"
              }`}
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>

          {/* Filter Dropdown */}
          {filterOpen && (
            <div
              className="mt-1 w-64 sm:w-80 bg-[#0a0a0a]/95 text-white rounded-md
                        shadow-lg p-2 sm:p-3 space-y-3 border border-white/10
                        backdrop-blur-md z-20"
            >
              {classes.length === 0 ? (
                <div className="text-center py-2 text-[10px] sm:text-xs text-gray-400">
                  No filters available
                </div>
              ) : (
                <>
                  {/* Header */}
                  <div className="flex items-center justify-between mb-1 sm:mb-2">
                    <h3 className="text-[10px] sm:text-[11px] uppercase tracking-wide text-gray-300">
                      Filter Detection Results
                    </h3>
                    <button
                      onClick={() => {
                        if (
                          filterRaw.selectedClasses.length === classes.length
                        ) {
                          setFilterRaw({ selectedClasses: [] });
                        } else {
                          setFilterRaw({ selectedClasses: [...classes] });
                        }
                      }}
                      className="text-[10px] sm:text-[11px] text-gray-300 underline hover:text-white transition border-0 bg-transparent p-0 cursor-pointer"
                    >
                      {filterRaw.selectedClasses.length === classes.length
                        ? "Unselect All"
                        : "Select All"}
                    </button>
                  </div>

                  <div className="border-b border-gray-700"></div>

                  {/* Grouped Filters */}
                  {(() => {
                    const treeClasses = classes.filter(
                      (c) =>
                        c.toLowerCase() === "tree" ||
                        c.toLowerCase().includes("trees-")
                    );

                    const cropClasses = classes.filter((c) =>
                      ["romaine", "lettuce", "potato", "bokchoy"].includes(
                        c.toLowerCase()
                      )
                    );

                    const renderGroup = (title: string, group: string[]) =>
                      group.length > 0 && (
                        <div className="mt-3">
                          <h4 className="text-[10px] sm:text-[11px] uppercase tracking-wide text-gray-400 mb-1">
                            {title}
                          </h4>
                          <div className="grid grid-cols-2 gap-1.5 sm:gap-2">
                            {group.map((cls) => (
                              <button
                                key={cls}
                                onClick={() => {
                                  setFilterRaw((prev) => {
                                    const selected =
                                      prev.selectedClasses.includes(cls)
                                        ? prev.selectedClasses.filter(
                                            (c) => c !== cls
                                          )
                                        : [...prev.selectedClasses, cls];
                                    return {
                                      ...prev,
                                      selectedClasses: selected,
                                    };
                                  });
                                }}
                                className={`min-w-[55px] sm:min-w-[65px] text-center px-1.5 sm:px-2.5 py-1.5 sm:py-1.75 rounded-md text-[10.5px] sm:text-[11px] border transition cursor-pointer ${
                                  filterRaw.selectedClasses.includes(cls)
                                    ? "bg-emerald-600 text-white border-emerald-600 hover:bg-emerald-700"
                                    : "bg-neutral-900 text-white border-white/10 hover:bg-neutral-800"
                                }`}
                              >
                                {cls}
                              </button>
                            ))}
                          </div>
                        </div>
                      );

                    return (
                      <>
                        {renderGroup("Trees", treeClasses)}
                        {renderGroup("Crops", cropClasses)}

                        {/* Confidence Filter */}
                        {detectionsLoaded && (
                          <div className="mt-4 pt-3 border-t border-gray-700">
                            <h4 className="text-[10px] sm:text-[11px] uppercase tracking-wide text-gray-400 mb-1.5">
                              Confidence Threshold
                            </h4>

                            <div className="flex items-center space-x-2">
                              <input
                                type="range"
                                min={0}
                                max={1}
                                step={0.01}
                                value={confidenceThresholdRaw}
                                onChange={(e) =>
                                  setConfidenceThresholdRaw(
                                    parseFloat(e.target.value)
                                  )
                                }
                                className="w-full accent-emerald-500 cursor-pointer"
                              />

                              <span className="text-[10px] sm:text-[11px] text-gray-300 w-8 text-right">
                                {(confidenceThresholdRaw * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        )}
                      </>
                    );
                  })()}
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Floating Legend */}
      <div
        className="absolute top-2 right-2 bg-[#0a0a0a]/95 text-white rounded-md shadow-lg
                  p-2 sm:p-3 backdrop-blur-md border border-white/10 z-20 min-w-[150px] sm:min-w-[180px]
                  max-w-[250px] sm:max-w-[300px]"
      >
        <div
          className={`flex items-center mb-1 sm:mb-2 ${
            Object.keys(legendCounts).length > 0 &&
            Object.values(visibleLayers).some((v) => v)
              ? "justify-between"
              : "justify-center"
          }`}
        >
          <h3
            className="text-[10px] sm:text-[11px] uppercase tracking-wide text-gray-300"
            data-qg="legend-panel"
          >
            Legend
          </h3>
          {Object.values(visibleLayers).some((v) => v) &&
            Object.values(legendCounts).some((count) => count > 0) && (
              <button
                onClick={() => setExpandedLegend(!expandedLegend)}
                className="text-[9px] sm:text-[10px] text-gray-400 hover:text-white transition cursor-pointer bg-transparent border-0"
              >
                {expandedLegend ? "Collapse" : "Expand"}
              </button>
            )}
        </div>

        <div className="border-b border-gray-700 mb-1 sm:mb-2"></div>

        {!mapReady || !legendReady || legendLoading ? (
          <div className="flex items-center justify-center gap-1 text-[9px] sm:text-[10px] text-gray-400">
            <span>
              {legendLoading ? "Updating legend" : "Loading detections"}
            </span>
            <Loader className="w-3 h-3 animate-spin text-gray-400" />
          </div>
        ) : Object.keys(legendCounts).length === 0 ? (
          <p className="text-gray-400 text-[9px] sm:text-[10px] text-center">
            No detections
          </p>
        ) : (
          <ul className="space-y-0.5 sm:space-y-1 text-[10px] sm:text-[11px]">
            {Object.entries(legendCounts).map(([cls, count]) => {
              let color = "";
              if (detectionsLayerRef.current) {
                const feature = detectionsLayerRef.current
                  .getSource()
                  ?.getFeatures()
                  .find((f) => f.get("label") === cls);
                if (feature) color = feature.get("color");
              }

              return (
                <li
                  key={cls}
                  className="bg-neutral-900/60 hover:bg-neutral-800/80 px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md transition"
                >
                  <div className="flex justify-between items-center">
                    <div className="flex items-center space-x-1.5 sm:space-x-2">
                      <span
                        className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full border border-white/20"
                        style={{ backgroundColor: color }}
                      ></span>
                      <span className="text-gray-200">{cls}</span>
                    </div>
                    <span className="text-gray-100 font-medium">{count}</span>
                  </div>

                  {expandedLegend && (
                    <ul className="mt-0.5 sm:mt-1 ml-4 sm:ml-5 space-y-0.5">
                      {Object.entries(landCoverBreakdown)
                        .filter(([lcName]) => visibleLayers[lcName])
                        .filter(([_, lcCounts]) => (lcCounts[cls] ?? 0) > 0)
                        .map(([lcName, lcCounts]) => {
                          const lcValue = lcCounts[cls];
                          return (
                            <li
                              key={lcName}
                              className="flex justify-between text-gray-400"
                            >
                              <span className="text-[9px] sm:text-[10px] italic">
                                {lcName}
                              </span>
                              <span className="text-[9px] sm:text-[10px]">
                                {lcValue}
                              </span>
                            </li>
                          );
                        })}
                    </ul>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Map */}
      <div
        ref={mapRef}
        className="fixed top-14 left-0 right-0 bottom-0 w-full h-[calc(100vh-3.5rem)]"
      />
    </div>
  );
}

const mongoose = require("mongoose");
const geojson = require("../../models/geojson-model");
const Detection = require("../../models/detection-model");

const colorMap = {
  "trees-5pma": "#0B6623",
  "Tree": "#0B6623",
  "Bokchoy": "#00BFA5",     
  "Potato": "#FFA500",
  "Lettuce": "#7CFC00",
  "Romaine": "#0A2F0A" 
};

//Get all geojson files
exports.getAllGeojson = async (req, res) => {
  try {
    const geojsonFiles = await geojson.find();
    res.json(geojsonFiles);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

//Get geojson by name
exports.getGeojsonByName = async (req, res) => {
  try {
    const geojsonFile = await geojson.findOne({ name: req.params.name });
    if (!geojsonFile) return res.status(404).json({ message: "Geojson not found" });
    res.json(geojsonFile);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

//Return all detections as GeoJSON
exports.getDetectionsGeoJSON = async (req, res) => {
  try {
    const { project_id } = req.query;
    if (!project_id) return res.status(400).json({ error: "project_id is required" });

    const projectObjectId = new mongoose.Types.ObjectId(project_id);

    const query = { project_id: projectObjectId };

    if (req.query.min_confidence) {
      query.confidence = { $gte: parseFloat(req.query.min_confidence) };
    }

    const detections = await Detection.find(query)
      .select("gps_coordinates label project_id date confidence -_id")
      .lean();

    const geojsonData = {
      type: "FeatureCollection",
      features: detections
        .filter(d => d.gps_coordinates)
        .map(d => ({
          type: "Feature",
          properties: {
            label: d.label,
            project_id: d.project_id,
            date: d.date,
            confidence: d.confidence,
            color: colorMap[d.label] || colorMap.default,
          },
          geometry: {
            type: "Point",
            coordinates: [d.gps_coordinates.lon, d.gps_coordinates.lat],
          },
        })),
    };

    res.json(geojsonData);
  } catch (err) {
    console.error("GeoJSON fetch error:", err);
    res.status(500).json({ error: "Failed to fetch detections" });
  }
};

//Get detection classes for a project
exports.getDetectionClasses = async (req, res) => {
  try {
    const { project_id } = req.query;
    const query = project_id
      ? { project_id: new mongoose.Types.ObjectId(project_id) }
      : {};
    const classes = await Detection.distinct("label", query);
    res.json(classes);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch detection classes" });
  }
};

//Get filtered detections as GeoJSON
exports.getFilteredDetections = async (req, res) => {
  try {
    const { project_id, labels } = req.query;
    if (!project_id) {
      return res.status(400).json({ error: "project_id is required" });
    }

    const projectObjectId = new mongoose.Types.ObjectId(project_id);
    const query = { project_id: projectObjectId };

    if (req.query.min_confidence) {
      query.confidence = { $gte: parseFloat(req.query.min_confidence) };
    }

    if (labels) {
      query.label = { $in: labels.split(",").map((l) => l.trim()) };
    }

    const detections = await Detection.find(query)
      .select("gps_coordinates label project_id date confidence -_id")
      .lean();

    const geojsonData = {
      type: "FeatureCollection",
      features: detections
        .filter((d) => d.gps_coordinates)
        .map((d) => ({
          type: "Feature",
          properties: {
            label: d.label,
            project_id: d.project_id,
            date: d.date,
            confidence: d.confidence,
            color: colorMap[d.label] || colorMap.default,
          },
          geometry: {
            type: "Point",
            coordinates: [d.gps_coordinates.lon, d.gps_coordinates.lat],
          },
        })),
    };
    res.json(geojsonData);
  } catch (err) {
    console.error("Filter error:", err);
    res.status(500).json({ error: "Failed to fetch detections" });
  }
};
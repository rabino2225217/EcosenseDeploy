const express = require('express');
const router = express.Router();
const activeLayerController = require('../../controllers/admin/wmsController');
const layerController = require("../../controllers/client/layerController");

//GET /layer/
router.get("/get", layerController.getAllGeojson);

//GET /layer/detections
router.get("/detections", layerController.getDetectionsGeoJSON);

//GET /layer/classes
router.get("/classes", layerController.getDetectionClasses); 

//GET /layer/filtered-detections
router.get("/filtered-detections", layerController.getFilteredDetections);

//GET /layer/active-layer
router.get("/active-layer", activeLayerController.getActiveLayer);

//GET /layer/:name
router.get("/:name", layerController.getGeojsonByName);

module.exports = router;
const express = require("express");
const router = express.Router();
const summaryController = require("../../controllers/client/summaryController");

//POST /summary/save
router.post("/save", summaryController.saveSummary);

//GET /summary/latest
router.get("/latest", summaryController.getSummary);

//GET /summary/export/detections/:project_id
router.get("/export/detections/:project_id", summaryController.exportDetailedDetections);

module.exports = router;
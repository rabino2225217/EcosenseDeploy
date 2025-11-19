const express = require('express');
const router = express.Router();
const wmsController = require('../../controllers/admin/wmsController');

//GET /capabilities
router.get('/capabilities', wmsController.getCapabilities);

//POST /save-layer
router.post('/save-layer', wmsController.saveActiveLayer);

//GET /active-layer
router.get("/active-layer", wmsController.getActiveLayer);

//GET /tiles
router.get("/tiles", wmsController.getWMSTile)

module.exports = router;
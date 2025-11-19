const mongoose = require("mongoose");

const geojsonSchema = new mongoose.Schema({}, { strict: false });

module.exports = mongoose.model("GeoJSON", geojsonSchema, "geojson");
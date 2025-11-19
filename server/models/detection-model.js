const mongoose = require("mongoose");

const DetectionSchema = new mongoose.Schema({
  project_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Project", 
    required: true,
  },
  label: String,
  bbox: {
    x1: Number,
    y1: Number,
    x2: Number,
    y2: Number,
  },
  gps_coordinates: {
    lat: Number,
    lon: Number,
  },
  confidence: Number,
  date: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Detection", DetectionSchema);
const mongoose = require("mongoose");

const SummarySchema = new mongoose.Schema({
  project_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Project",
    required: true,
    unique: true,
  },
  land_covers: [
    {
      name: String,
      counts: {
        type: Map,
        of: Number,
      },
    },
  ],
  filters: [String],
  recorded_at: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model("Summary", SummarySchema, "summary");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const geojson = require("../../models/geojson-model");
const Summary = require("../../models/summary-model");

const CONVERSION_API_URL = process.env.CONVERSION_API_URL;

//Add Land Cover
exports.addLandCover = async (req, res) => {
  const io = req.app.get("io");
  const tempFile = req.file?.path; 
  
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded." });

    const { type, name } = req.body;

    if (!type || !["trees", "crops"].includes(type.toLowerCase())) {
      return res.status(400).json({ error: "Type must be 'trees' or 'crops'." });
    }

    if (!name || !name.trim()) {
      return res.status(400).json({ error: "Layer name is required." });
    }

    const trimmedName = name.trim();

    //Check for existing name
    const existing = await geojson.findOne({
      name: { $regex: new RegExp(`^${trimmedName}$`, "i") },
    });
    if (existing) {
      return res.status(400).json({ error: `Layer name already exists.` });
    }

    //Check for maximum limit (8 layers)
    const layerCount = await geojson.countDocuments();
    if (layerCount >= 8) {
      return res.status(400).json({
        error: "Maximum of 8 land cover layers reached. Please delete an existing layer before adding a new one.",
      });
    }

    //Prepare file stream for Flask conversion API
    const formData = new FormData();
    formData.append("file", fs.createReadStream(tempFile));
    formData.append("type", type.toLowerCase());

    //Call Conversion API
    const response = await axios.post(CONVERSION_API_URL, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 60000,
    });

    //Prepare data to save
    const geoData = {
      type: "FeatureCollection",
      name: trimmedName,
      land_type: type.toLowerCase(),
      date_added: new Date(),
      crs:
        response.data.crs || {
          type: "name",
          properties: { name: "urn:ogc:def:crs:OGC:1.3:CRS84" },
        },
      features: response.data.features,
    };

    //Save to DB
    const newLayer = await geojson.create(geoData);

    io.emit("landcover:update", { action: "add", layer: newLayer });
    return res.status(201).json({
      message: "Land cover layer added successfully.",
      layer: newLayer,
    });
  } catch (error) {
    console.error("Error adding land cover:", error.message);
    const flaskError =
      error.response?.data?.error ||
      "Invalid GeoTIFF file or conversion failed.";
    return res.status(400).json({ error: flaskError });
  } finally {
    if (tempFile && fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile);
    }
  }
};

//Rename Land Cover
exports.updateLandCover = async (req, res) => {
  const io = req.app.get("io");
  try {
    const { id } = req.params;
    const { name } = req.body;

    if (!name || !name.trim()) {
      return res.status(400).json({ error: "Layer name is required." });
    }

    const trimmedName = name.trim();

    const existing = await geojson.findOne({
      _id: { $ne: id },
      name: { $regex: new RegExp(`^${trimmedName}$`, "i") },
    });

    if (existing) {
      return res.status(400).json({
        error: "A layer with this name already exists. Please choose a different name."
      });
    }

    const oldLayer = await geojson.findById(id);
    if (!oldLayer) {
      return res.status(404).json({ error: "Layer not found." });
    }

    const updated = await geojson.findByIdAndUpdate(
      id,
      { name: trimmedName },
      { new: true }
    );

    await Summary.updateMany(
      { "land_covers.name": oldLayer.name },
      { $set: { "land_covers.$[elem].name": trimmedName } },
      { arrayFilters: [{ "elem.name": oldLayer.name }] }
    );

    io.emit("landcover:update", {
      action: "rename",
      layer: {
        ...updated.toObject(),
        oldName: oldLayer.name, 
      },
    });

    res.status(200).json({  
      message: "Layer renamed successfully.",
      layer: updated,
    });
  } catch (error) {
    console.error("Error renaming layer:", error);
    res.status(500).json({ error: "Server error while renaming layer." });
  }
};

//Delete Land Cover
exports.deleteLandCover = async (req, res) => {
  const io = req.app.get("io");
  try {
    const { id } = req.params;

    const deleted = await geojson.findByIdAndDelete(id);
    if (!deleted) return res.status(404).send("Layer not found.");

    await Summary.updateMany(
      {},
      { $pull: { land_covers: { name: deleted.name } } }
    );

    io.emit("landcover:update", {
      action: "delete",
      layer: {
        _id: deleted._id,
        name: deleted.name
      },
    });
    res.status(200).json({ message: "Layer deleted successfully." });
  } catch (error) {
    console.error("Error deleting layer:", error);
    res.status(500).send("Server error while deleting layer.");
  }
};
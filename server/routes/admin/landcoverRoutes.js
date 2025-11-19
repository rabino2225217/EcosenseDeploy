const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const router = express.Router();
const landcoverController = require("../../controllers/admin/landcoverController");

//Uploads directory
const uploadDir = path.join(__dirname, "../../uploads/temp");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

//Disk-based storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const unique = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, unique + path.extname(file.originalname));
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 15 * 1024 * 1024 }, 
});

//POST /add
router.post("/add", upload.single("file"), landcoverController.addLandCover);

//PUT /update/:id
router.put("/update/:id", landcoverController.updateLandCover);

//DELETE /delete/:id
router.delete("/delete/:id", landcoverController.deleteLandCover);

module.exports = router;
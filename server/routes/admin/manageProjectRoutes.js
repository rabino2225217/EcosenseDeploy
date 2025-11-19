const express = require("express");
const router = express.Router();
const projectController = require("../../controllers/admin/manageProjectController");

//GET /all
router.get("/all", projectController.getAllProjects);

//PUT /rename/:id
router.put("/:id/rename", projectController.adminRenameProject);

//DELETE /delete/:id
router.delete("/:id/delete", projectController.adminDeleteProject);

module.exports = router;
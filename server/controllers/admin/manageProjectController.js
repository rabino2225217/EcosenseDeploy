const mongoose = require("mongoose");
const Project = require("../../models/project-model");
const Detection = require("../../models/detection-model");
const Summary = require("../../models/summary-model");

//Fetch all projects for the admin
exports.getAllProjects = async (req, res) => {
  const userId = req.session.userId;

  if (!userId) {
    return res.status(401).json({ message: "Unauthorized" });
  }

  try {
    const projects = await Project.find()
      .populate("userId", "name")
      .select("name createdAt userId")
      .sort({ createdAt: -1 });

    return res.status(200).json({message: "All projects fetched successfully.", data: projects,});
  } catch (error) {
    console.error("Error fetching all projects:", error);
    return res.status(500).json({message: "Failed to fetch projects.",error: error.message,});
  }
};

//Rename project
exports.adminRenameProject = async (req, res) => {
  const io = req.app.get("io");
  const { id } = req.params;
  const { name } = req.body;
  const userId = req.session.userId;

  if (!userId) {
    return res.status(401).json({ message: "Unauthorized" });
  }

  if (!name || !name.trim()) {
    return res.status(400).json({ message: "Project name is required." });
  }

  const trimmedName = name.trim().toLowerCase();

  try {
    const project = await Project.findById(id);
    if (!project) {
      return res.status(404).json({ message: "Project not found." });
    }

    const duplicate = await Project.findOne({
      userId: project.userId,
      _id: { $ne: id },
      $expr: {
        $eq: [{ $toLower: "$name" }, trimmedName],
      },
    });

    if (duplicate) {
      return res.status(400).json({
        message: "This user already has a project with that name.",
      });
    }

    project.name = name.trim();
    await project.save();

    io.to("admins").emit("project:renamed", { id, name });
    io.to(project.userId.toString()).emit("project:renamed", { id, name });
    return res.status(200).json({message: "Project renamed successfully.", project,});
  } catch (error) {
    console.error("Error renaming project:", error);
    return res.status(500).json({
      message: "Failed to rename project.",
      error: error.message,
    });
  }
};

//Delete project
exports.adminDeleteProject = async (req, res) => {
  const io = req.app.get("io");
  const userId = req.session.userId;
  const { id } = req.params;

  if (!userId) {
    return res.status(401).json({ message: "Unauthorized" });
  }

  try {
    const project = await Project.findById(id);
    if (!project) {
      return res.status(404).json({ message: "Project not found." });
    }

    const name = project.name;
    const projectObjectId = new mongoose.Types.ObjectId(id);

    await Promise.all([
      Detection.deleteMany({ project_id: projectObjectId }),
      Summary.deleteMany({ project_id: projectObjectId }),
      Project.findByIdAndDelete(projectObjectId),
    ]);

    io.to("admins").emit("project:deleted", { id, name });
    io.to(project.userId.toString()).emit("project:deleted", { id, name });

    return res.status(200).json({
      message: "Project and related data deleted successfully.",
      deletedId: id,
    });
  } catch (error) {
    console.error("Error deleting project and related data:", error);
    return res.status(500).json({
      message: "Failed to delete project and related data.",
      error: error.message,
    });
  }
};
const mongoose = require("mongoose");
const User = require('../../models/user-model')
const Project = require('../../models/project-model');
const Detection = require('../../models/detection-model');
const Summary = require("../../models/summary-model");

//Create a new project
exports.createProject = async (req, res) => {
  const io = req.app.get("io");
  const userId = req.session.userId;

  let { name, description, location } = req.body;

  if (!userId) {
    return res.status(401).json({ message: "Unauthorized" });
  }

  try {
    const user = await User.findById(userId).select("role");
    if (!user) {
      return res.status(404).json({ message: "User not found." });
    }

  
    if (user.role === "Admin") {
      return res.status(403).json({ message: "Admins cannot create projects as clients." });
    }

    name = name?.trim();
    location = location?.trim();

    if (!name || !location) {
      return res.status(400).json({ message: "Project name and location are required." });
    }

    const existing = await Project.findOne({
      userId,
      name: { $regex: new RegExp(`^${name}$`, "i") },
    });

    if (existing) {
      return res.status(400).json({ message: "You already have a project with this name." });
    }

    const newProject = new Project({ name, description, location, userId });
    await newProject.save();

    const populatedProject = await newProject.populate("userId", "name");

    io.to("admins").emit("project:created", populatedProject);
    io.to(userId).emit("project:created", populatedProject);

    return res.status(201).json({project: newProject, message: "Project created successfully!",});
  } catch (err) {
    console.error("Error creating project:", err);
    return res.status(500).json({ message: "Unable to create project. Please try again later." });
  }
};

//Fetch projects for the logged-in user
exports.listProjects = async (req, res) => {
  const userId = req.session.userId;
  if (!userId) return res.status(401).json({ message: "Unauthorized" });

  const search = (req.query.search || "").trim();

  try {
    const query = {
      userId,
      name: { $regex: new RegExp(search, "i") },
    };
    const projects = await Project.find(query).sort({ createdAt: -1 });
    return res.status(200).json({ message: "Projects fetched successfully", data: projects });
  } catch (err) {
    console.error("Error fetching projects:", err);
    return res.status(500).json({ message: "Failed to fetch projects" });
  }
};

//Get project by ID
exports.getProjectById = async (req, res) => {
  const userId = req.session.userId;
  const { id } = req.params;
  if (!userId) return res.status(401).json({ message: "Unauthorized" });

  try {
    const project = await Project.findOne({ _id: id, userId });
    if (!project) return res.status(404).json({ message: "Project not found" });
    return res.status(200).json({ message: "Project fetched successfully", data: project });
  } catch (err) {
    console.error("Error fetching project:", err);
    return res.status(500).json({ message: "Failed to fetch project" });
  }
};

//Rename project
exports.renameProject = async (req, res) => {
  const io = req.app.get('io');
  const userId = req.session.userId;
  const { id } = req.params;
  let { name } = req.body;

  if (!userId) return res.status(401).json({ message: "Unauthorized" });

  name = name?.trim();
  if (!name) return res.status(400).json({ message: "Project name is required." });

  try {
    const project = await Project.findOne({ _id: id, userId });
    if (!project) return res.status(404).json({ message: "Project not found." });

    const duplicate = await Project.findOne({
      userId,
      _id: { $ne: id },
      name: { $regex: new RegExp(`^${name}$`, 'i') },
    });
    if (duplicate) {
      return res.status(400).json({ message: "You already have a project with this name." });
    }

    project.name = name;
    await project.save();

    io.to("admins").emit("project:renamed", { id, name });
    io.to(userId).emit("project:renamed", { id, name });
    return res.status(200).json({ project, message: "Project renamed successfully!" });
  } catch (err) {
    console.error("Error renaming project:", err);
    return res.status(500).json({ message: "Failed to rename project." });
  }
};

//Delete project
exports.deleteProject = async (req, res) => {
  const io = req.app.get("io");
  const userId = req.session.userId;
  const { id } = req.params;

  if (!userId) return res.status(401).json({ message: "Unauthorized" });

  try {
    const project = await Project.findOneAndDelete({ _id: id, userId });
    if (!project) return res.status(404).json({ message: "Project not found." });

    const projectObjectId = new mongoose.Types.ObjectId(id);

    await Promise.all([
      Detection.deleteMany({ project_id: projectObjectId }),
      Summary.deleteMany({ project_id: projectObjectId }),
    ]);

    io.to("admins").emit("project:deleted", { id });
    io.to(userId).emit("project:deleted", { id });

    return res.status(200).json({ message: "Project deleted successfully!" });
  } catch (err) {
    console.error("Error deleting project and related data:", err);
    return res.status(500).json({ message: "Failed to delete project and related data." });
  }
};
  const mongoose = require("mongoose");
  const User = require('../../models/user-model');
  const Project = require("../../models/project-model");
  const Detection = require("../../models/detection-model");
  const Summary = require("../../models/summary-model");
  const bcrypt = require('bcrypt');

  const emailRegex = /^(?!.*\.\.)[a-zA-Z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,10}$/;
  const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,20}$/;
  const nameRegex = /^[a-zA-ZÑñ\s'-]{2,50}$/;

  //Get users by status
  exports.getUsers = async (req, res) => {
    try {
      const searchQuery = req.query.search || '';
      const status = req.query.status; 

      const query = { role: 'DENR staff' }; 
      
      if (status) {
        query.status = status; 
      }

      if (searchQuery) {
        query.name = { $regex: searchQuery, $options: 'i' };
      }

      const users = await User.find(query, '-password');
      res.status(200).send(users);
    } catch (error) {
      console.error("Error fetching users:", error);
      res.status(500).send("Server error");
    }
  };

  //Add new user
  exports.addUser = async (req, res) => {
    const io = req.app.get('io');
    try {
      let { name, email, password } = req.body;

      name = name?.trim();
      email = email?.trim()?.toLowerCase();

      if (!name || !email || !password) {
        return res.status(400).json({ message: "All fields are required." });
      }

      if (!nameRegex.test(name)) {
        return res.status(400).json({ message: "Invalid name format." });
      }
      
      if (!emailRegex.test(email)) {
        return res.status(400).json({ message: "Invalid email format." });
      }

      if (email.length > 100) {
        return res.status(400).send("Email must not exceed 100 characters.");
      }

      const existing = await User.findOne({ email });
      if (existing) {
        return res.status(409).json({ message: "User already exists." });
      }

      const hashedPassword = await bcrypt.hash(password, 10);

      const newUser = new User({
        name,
        email,
        password: hashedPassword,
        role: "DENR staff",
        isActive: true,
        status: "active",
      });

      await newUser.save();
      const { password: _, ...userWithoutPassword } = newUser.toObject();

      io.to("admins").emit("user:added", userWithoutPassword);
      res.status(201).json({ message: "User created successfully!", user: userWithoutPassword });
    } catch (error) {
      console.error("Error adding user:", error);
      res.status(500).json({ message: "Server error." });
    }
  };

  //Update user details
  exports.updateUserDetails = async (req, res) => {
    const io = req.app.get('io');
    const { id } = req.params;
    try {
      let { name, email } = req.body;

      name = name?.trim();
      email = email?.trim()?.toLowerCase();

      if (!name || !email) {
        return res.status(400).send("Name and email are required.");
      }

      if (!nameRegex.test(name)) {
        return res.status(400).send("Invalid name format.");
      }

      if (!emailRegex.test(email)) {
        return res.status(400).send("Invalid email format.");
      }

      if (email.length > 100) {
        return res.status(400).send("Email must not exceed 100 characters.");
      }

      const existing = await User.findOne({ email, _id: { $ne: id } });
      if (existing) {
        return res.status(409).send("Email is already in use by another user.");
      }

      const updatedUser = await User.findByIdAndUpdate(
        id,
        { name, email },
        { new: true, select: "-password" }
      );

      if (!updatedUser) return res.status(404).send("User not found.");

      io.to("admins").emit("user:updated", updatedUser);
      io.to(updatedUser._id.toString()).emit("user:updated", updatedUser);
      res.send("User details updated.");
    } catch (error) {
      console.error("Error updating user details:", error);
      res.status(500).send("Server error.");
    }
  };

  //Update user password
  exports.updatePassword = async (req, res) => {
    const io = req.app.get("io");
    const { id } = req.params;

    try {
      let { password } = req.body;
      if (!password) return res.status(400).send("Password required.");

      const hashed = await bcrypt.hash(password, 10);
      const updatedUser = await User.findByIdAndUpdate(
        id,
        { password: hashed },
        { new: true, select: "-password" }
      );

      if (!updatedUser) return res.status(404).send("User not found.");

      try {
        const db = mongoose.connection.db;
        const sessions = db.collection("sessions");

        await sessions.deleteMany({
          session: { $regex: `"userId":"${id}"` },
        });

        console.log(`Sessions for user ${id} deleted after password reset.`);
      } catch (sessionErr) {
        console.error("Error deleting user sessions:", sessionErr);
      }

      io.to("admins").emit("user:passwordUpdated", { _id: id });
      io.to(id).emit("user:forceLogout"); 

      res.send("Password updated and user logged out successfully.");
    } catch (error) {
      console.error("Error updating password:", error);
      res.status(500).send("Server error.");
    }
  };

  //Toggle user active status
  exports.toggleActive = async (req, res) => {
    const io = req.app.get("io");
    const { id } = req.params;
    const { isActive } = req.body;

    try {
      const updatedUser = await User.findByIdAndUpdate(
        id,
        { isActive },
        { new: true, select: "-password" }
      );

      if (!updatedUser) return res.status(404).send("User not found.");

      io.to("admins").emit("user:toggled", updatedUser);

      if (isActive === false) {
        try {
          const db = mongoose.connection.db;
          const sessions = db.collection("sessions");

          await sessions.deleteMany({
            session: { $regex: `"userId":"${id}"` },
          });

          console.log(`Sessions for user ${id} deleted after deactivation.`);
        } catch (sessionErr) {
          console.error("Error deleting user sessions:", sessionErr);
        }

        io.to(id).emit("user:deactivated");
      }

      res.status(200).send(updatedUser);
    } catch (error) {
      console.error("Error toggling active state:", error);
      res.status(500).send("Server error.");
    }
  };

  //Delete user
  exports.deleteUser = async (req, res) => {
    const io = req.app.get("io");
    const { id } = req.params;

    try {
      const userProjects = await Project.find({ userId: id });

      if (userProjects.length) {
        const projectIds = userProjects.map((p) => new mongoose.Types.ObjectId(p._id));

        await Promise.all([
          Detection.deleteMany({ project_id: { $in: projectIds } }),
          Summary.deleteMany({ project_id: { $in: projectIds } }),
          Project.deleteMany({ userId: id }),
        ]);

        projectIds.forEach((projectId) => {
          io.to("admins").emit("project:deleted", { id: projectId.toString() });
        });
      }

      try {
        const db = mongoose.connection.db;
        const sessionsCollection = db.collection("sessions");

        await sessionsCollection.deleteMany({
          session: { $regex: `"userId":"${id}"` },
        });
      } catch (err) {
        console.error("Error deleting sessions:", err);
      }

      await User.findByIdAndDelete(id);
      io.emit("user:deleted", { _id: id });

      res.send(
        userProjects.length
          ? "User and related data deleted successfully."
          : "User deleted. No related projects found."
      );
    } catch (error) {
      console.error("Error deleting user and related data:", error);
      res.status(500).send("Server error.");
    }
  };

  //Approve user
  exports.approveUser = async (req, res) => {
    const io = req.app.get('io');
    const { id } = req.params;

    try {
      const user = await User.findById(id);
      if (!user) return res.status(404).send("User not found.");
      if (user.status !== 'pending') return res.status(400).send("User is not pending.");

      user.status = 'active';
      user.isActive = true;
      await user.save();

      io.to("admins").emit("user:approved", user);
      res.status(200).send({ message: "User approved.", user });
    } catch (err) {
      console.error(err);
      res.status(500).send("Server error.");
    }
  };

  //Reject user
  exports.rejectUser = async (req, res) => {
    const io = req.app.get('io');
    const { id } = req.params;

    try {
      const user = await User.findById(id);
      if (!user) return res.status(404).send("User not found.");
      if (user.status !== 'pending') return res.status(400).send("User is not pending.");
      
      await user.deleteOne();

      io.to("admins").emit("user:rejected", { _id: id });
      res.status(200).send("User rejected and removed.");
    } catch (err) {
      console.error(err);
      res.status(500).send("Server error.");
    }
  };
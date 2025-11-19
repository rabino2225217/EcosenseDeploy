const User = require('../../models/user-model');
const bcrypt = require('bcrypt');
const mongoose = require('mongoose');

const emailRegex = /^(?!.*\.\.)[a-zA-Z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,10}$/;
const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,20}$/;
const nameRegex = /^[a-zA-ZÑñ\s'-]{2,50}$/;

//Register
exports.registerUser = async (req, res) => {
  let { email, name, password } = req.body;

  name = name?.trim();
  email = email?.trim()?.toLowerCase();

  if (!email || !name || !password) {
    return res.status(400).send("All fields are required.");
  }

  if (!nameRegex.test(name)) {
    return res.status(400).send("Name must be 2-50 characters and only letters, spaces, apostrophes, or hyphens.");
  }

  if (!emailRegex.test(email)) {
    return res.status(400).send("Please enter a valid email address.");
  }

  if (email.length > 100) {
    return res.status(400).send("Email must not exceed 100 characters.");
  }

  if (!passwordRegex.test(password)) {
    return res.status(400).send("Password must be 8-20 characters, include uppercase, lowercase, number, and special character.");
  }

  try {
    const existing = await User.findOne({ email });
    if (existing) return res.status(409).send("Registration failed. Please try again.");

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({
      email,
      name,
      password: hashedPassword,
      role: 'DENR staff',
      isActive: false,
      status: 'pending'
    });

    await newUser.save();

    const io = req.app.get("io");

    io.to("admins").emit("user:added", {
      _id: newUser._id,
      name: newUser.name,
      email: newUser.email,
      status: newUser.status,
      isActive: newUser.isActive,
    });

    res.status(201).send("Registration successful! Please wait for admin approval.");
  } catch (err) {
    console.error(err);
    res.status(500).send("Registration failed. Please try again.");
  }
};

//Login
exports.loginUser = async (req, res) => {
  let { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).send("Email and password are required.");
  }

  try {
    email = email.toLowerCase();

    const user = await User.findOne({ email });
    if (!user) return res.status(401).send("Invalid credentials.");

    if (!user.isActive) {
      if (user.status === "pending") {
        return res.status(403).send("This account is pending approval by the administrator.");
      }
      return res.status(403).send("This account is inactive. Please contact the administrator.");
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(401).send("Invalid credentials.");

    //Check if user already has an active session
    const db = mongoose.connection.db;
    const sessions = db.collection("sessions");
    const activeSession = await sessions.findOne({
      "session": { $regex: `"userId":"${user._id}"` }
    });

    if (activeSession) {
      return res.status(403).send("Unable to login at this time.");
    }

    //Create session
    req.session.userId = user._id;
    req.session.role = user.role;

    req.session.save((err) => {
      if (err) {
        console.error("Session save error:", err);
        return res.status(500).send("Login failed. Please try again.");
      }

      //Redirect based on role
      let redirectUrl;
      switch (user.role) {
        case "DENR staff":
          redirectUrl = "/app";
          break;
        case "Admin":
          redirectUrl = "/admin";
          break;
        default:
          return res.status(403).send("Unauthorized role.");
      }

      return res.status(200).json({ redirect: redirectUrl });
    });
  } catch (err) {
    console.error(err);
    res.status(500).send("Login failed. Please try again.");
  }
};

//Logout
exports.logoutUser = async (req, res) => {
  try {
    req.session.destroy(err => {
      if (err) {
        return res.status(500).send("An error occurred while logging out.");
      }
      res.clearCookie('connect.sid');
      res.send("Logged out successfully.");
    });
  } catch (err) {
    console.error(err);
    res.status(500).send("Logout failed. Please try again.");
  }
};

//Get logged in user info
exports.getCurrentUser = async (req, res) => {
  if (!req.session.userId) {
    return res.status(401).send("Not logged in");
  }

  try {
    const user = await User.findById(req.session.userId).select("name email role");
    if (!user) return res.status(404).send("Invalid session");

    res.json(user);
  } catch (err) {
    console.error(err);
    res.status(500).send("Server error.");
  }
};
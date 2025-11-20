require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const MongoStore = require("connect-mongo");
const cors = require('cors');
const path = require('path');
const fs = require('fs'); 
const session = require('express-session');
const http = require('http');
const compression = require("compression");
const { Server } = require('socket.io');
const sharedSession = require("express-socket.io-session");

const app = express();

// IMPORTANT: Trust proxy for Vercel (handles X-Forwarded-* headers correctly)
app.set("trust proxy", 1);

//Load ENV variables
const host = process.env.HOST || '0.0.0.0';
const port = Number(process.env.PORT) || 4000;
const mongodb = process.env.MONGODB_URI;
const isVercel = process.env.VERCEL === '1';
const isProduction = process.env.NODE_ENV === 'production';

// Parse CLIENT_ORIGIN (can be comma-separated for multiple origins)
const clientOrigin = process.env.CLIENT_ORIGIN || 'http://localhost:5173';

// Allowed origins list
const allowedOrigins = clientOrigin
  .split(',')
  .map(o => o.trim())
  .filter(Boolean);

if (!mongodb) {
  throw new Error("MONGODB_URI not set in environment variables");
}

console.log('Allowed origins:', allowedOrigins);


  // Fix for Vercel CORS
  app.use((req, res, next) => {
    const origin = req.headers.origin;
  
    if (allowedOrigins.includes(origin)) {
      res.setHeader("Access-Control-Allow-Origin", origin);
    }
  
    res.setHeader("Access-Control-Allow-Credentials", "true");
    res.setHeader("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  
    if (req.method === "OPTIONS") {
      return res.status(200).end();
    }
  
    next();
  });
//Middleware - Simple and secure CORS (matches working code pattern)
// app.use(cors({
//   origin: allowedOrigins,
//   credentials: true
// }));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: false, limit: '50mb' }));
app.use(compression());

//Session config
const sessionMiddleware = session({
  secret: process.env.SESSION_SECRET || 'ecosense-scrt-ky',
  resave: false,
  saveUninitialized: false,
  store: MongoStore.create({ mongoUrl: mongodb }),
  cookie: {
    secure: isProduction, // HTTPS in production (Vercel uses HTTPS)
    httpOnly: true,
    sameSite: isProduction ? 'none' : 'lax', // Required for cross-origin in production
    maxAge: 1000 * 60 * 60 * 24, // 24 hours
  },
});

app.use(sessionMiddleware);

//MongoDB connection
mongoose.connect(mongodb)
.then(() => console.log("Connected to MongoDB"))
.catch(err => console.error("MongoDB error:", err));

// Socket.IO setup (only for non-Vercel deployments)
let server, io;
if (!isVercel) {
  // Create HTTP server
  server = http.createServer(app);

  // Socket.IO setup
  io = new Server(server, {
    cors: {
      origin: allowedOrigins,
      credentials: true
    },
    transports: ["websocket"],
  });

  io.use(sharedSession(sessionMiddleware, {
    autoSave: true,
  }));

  io.on("connection", (socket) => {
    const session = socket.handshake.session;
    const role = session?.role;
    const userId = session?.userId;

    if (!userId || !role) {
      console.log("Unknown or unauthenticated socket:", socket.id);
      return;
    }

    if (role === "Admin") {
      socket.join("admins");
      console.log(`Admin joined: ${socket.id}`);
    } else {
      socket.join(userId.toString());
      console.log(`Client joined with userId ${userId}: ${socket.id}`);
    }

    socket.on("disconnect", () => {
      console.log(`Socket disconnected: ${socket.id}`);
    });
  });

  //Make io accessible in routes/controllers
  app.set('io', io);
} else {
  // For Vercel, create a mock io object to prevent errors
  app.set('io', {
    to: () => ({ emit: () => {} }),
    emit: () => {},
    in: () => ({ emit: () => {} })
  });
}

//Admin routes
app.use('/admin/users', require('./routes/admin/userRoutes'));
app.use('/admin/projects', require('./routes/admin/manageProjectRoutes'));
app.use('/admin/wms', require('./routes/admin/wmsRoutes'));
app.use('/admin/landcover', require('./routes/admin/landcoverRoutes'));

//Client routes
app.use('/analysis', require('./routes/client/analysisRoutes'));
app.use('/layer', require('./routes/client/layerRoutes'));
app.use('/auth', require('./routes/client/authRoutes'));
app.use('/project', require('./routes/client/projectRoutes'));
app.use('/summary', require('./routes/client/summaryRoutes'));

// Frontend is served by its own container/service, no need to serve static files here

//Auto-clean old temp files (only for non-Vercel deployments)
const tempPath = path.join(__dirname, "uploads/temp");

if (!isVercel && !fs.existsSync(tempPath)) {
  fs.mkdirSync(tempPath, { recursive: true });
}

const cleanTempFiles = () => {
  if (isVercel) return; // Skip on Vercel (serverless, no persistent storage)
  
  fs.readdir(tempPath, (err, files) => {
    if (err) return; 
    for (const file of files) {
      const filePath = path.join(tempPath, file);
      fs.stat(filePath, (err, stats) => {
        if (!err && Date.now() - stats.mtimeMs > 3 * 60 * 60 * 1000) {
          fs.unlink(filePath, (unlinkErr) => {
            if (!unlinkErr) {
              console.log(`Deleted old temp file: ${file}`);
            }
          });
        }
      });
    }
  });
};

//Run cleanup immediately once at startup and every 3 hours (only for non-Vercel)
if (!isVercel) {
  cleanTempFiles();
  setInterval(cleanTempFiles, 3 * 60 * 60 * 1000);
  
  //Start server
  server.listen(port, host, () => {
    console.log(`Server running at http://${host}:${port}`);
    console.log(`Allowed origins: ${allowedOrigins.join(", ")}`);
  });
} else {
  console.log('Running on Vercel - serverless mode');
  console.log(`Allowed origins: ${allowedOrigins.join(", ")}`);
}

// Export for Vercel
module.exports = app;
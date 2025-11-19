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

//Load ENV variables
const host = process.env.HOST || '0.0.0.0';
const port = process.env.PORT || 4000;
const mongodb = process.env.MONGODB_URI;
const isVercel = process.env.VERCEL === '1';
const clientOrigin = process.env.CLIENT_ORIGIN || (isVercel ? process.env.VERCEL_URL : `http://${host}:5173`);

// CORS configuration - support multiple origins
const allowedOrigins = process.env.ALLOWED_ORIGINS 
  ? process.env.ALLOWED_ORIGINS.split(',').map(o => o.trim())
  : [clientOrigin, `http://${host}:5173`, 'http://localhost:5173'];

//Middleware - CORS configuration
app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) {
      return callback(null, true);
    }
    
    // Always allow the frontend origin explicitly
    if (origin === 'https://ecosense-app.vercel.app') {
      return callback(null, true);
    }
    
    // Check if origin is in allowed list
    const isAllowed = allowedOrigins.some(allowed => {
      return origin === allowed || origin.includes(allowed) || allowed.includes(origin);
    });
    
    if (isAllowed) {
      callback(null, true);
    } else {
      // Log for debugging - but allow for now
      console.log(`CORS: Allowing origin ${origin} (allowed: ${allowedOrigins.join(', ')})`);
      callback(null, true);
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin'],
  exposedHeaders: ['Content-Type'],
  preflightContinue: false,
  optionsSuccessStatus: 204
}));

// Explicitly handle OPTIONS preflight requests
app.options('*', (req, res) => {
  res.header('Access-Control-Allow-Origin', req.headers.origin || '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS, PATCH');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Origin');
  res.header('Access-Control-Allow-Credentials', 'true');
  res.sendStatus(204);
});
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
    secure: isVercel || process.env.NODE_ENV === 'production', // HTTPS in production
    httpOnly: true,
    maxAge: 1000 * 60 * 60 * 24,
    sameSite: isVercel ? 'none' : 'lax', // Required for cross-origin in Vercel
  },
});

app.use(sessionMiddleware);

//MongoDB connection
if (mongodb) {
  mongoose.connect(mongodb)
  .then(() => console.log("Connected to MongoDB"))
  .catch(err => console.error("MongoDB error:", err));
}

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

// Health check endpoint for debugging
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    allowedOrigins: allowedOrigins,
    clientOrigin: clientOrigin,
    isVercel: isVercel,
    requestOrigin: req.headers.origin
  });
});

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

//Serve frontend build
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '..', 'client', 'dist')));
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'client', 'dist', 'index.html'));
  });
}

//Auto-clean old temp files
const tempPath = path.join(__dirname, "uploads/temp");

if (!fs.existsSync(tempPath)) {
  fs.mkdirSync(tempPath, { recursive: true });
}

const cleanTempFiles = () => {
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
  });
} else {
  console.log('Running on Vercel - serverless mode');
}

// Export for Vercel
module.exports = app;
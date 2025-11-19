require('dotenv').config();
const fs = require("fs").promises;
const LRU = require("lru-cache");

const ACTIVE_LAYER_FILE = process.env.ACTIVE_LAYER_FILE;
const COPERNICUS_ENDPOINT = process.env.COPERNICUS_ENDPOINT;

if (!COPERNICUS_ENDPOINT) throw new Error("COPERNICUS_ENDPOINT not set in .env!");

const LAYER_NAMES = [
  "AGRICULTURE",
  "ATMOSPHERIC_PENETRATION",
  "BATHYMETRIC",
  "COLOR_INFRARED",
  "COLOR_INFRARED__URBAN_",
  "GEOLOGY",
  "MOISTURE_INDEX",
  "TRUE_COLOR",
  "SWIR",
  "VEGETATION_INDEX",
];

//Tile cache 7 days
const TILE_CACHE_TTL = 7 * 24 * 60 * 60 * 1000;
const tileCache = new LRU({
  max: 500, 
  ttl: TILE_CACHE_TTL,
});

const inFlightRequests = new Map();

//Get available layers
exports.getCapabilities = (req, res) => {
  try {
    const allLayers = LAYER_NAMES.map((name) => ({ name }));
    res.json({ layers: allLayers });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch layers" });
  }
};

//Save active layer
exports.saveActiveLayer = async (req, res) => {
  const io = req.app.get("io");
  try {
    const { layerName } = req.body;
    if (!layerName) return res.status(400).json({ error: "layerName required" });

    const data = { layerName, updatedAt: new Date() };
    await fs.writeFile(ACTIVE_LAYER_FILE, JSON.stringify(data, null, 2));

    const responseData = { ...data, endpoint: COPERNICUS_ENDPOINT };
    io.emit("active-layer-updated", responseData);
    res.json(responseData);

    setTimeout(() => {
      try {
        tileCache.clear();
        inFlightRequests.clear();
        console.log("[WMS] Cleared tile + in-flight cache after layer switch");
      } catch (cacheErr) {
        console.warn("[WMS] Failed to clear cache:", cacheErr);
      }
    }, 2000);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to save active layer" });
  }
};

//Get active layer
exports.getActiveLayer = async (req, res) => {
  try {
    try {
      await fs.access(ACTIVE_LAYER_FILE);
    } catch {
      return res.status(404).json({ error: "No active layer set" });
    }

    const raw = await fs.readFile(ACTIVE_LAYER_FILE, "utf8");
    const data = JSON.parse(raw);

    res.json({ ...data, endpoint: COPERNICUS_ENDPOINT });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to read active layer" });
  }
};

//Cache Copernicus Tile
exports.getActiveLayer = async (req, res) => {
  try {
    await fs.access(ACTIVE_LAYER_FILE);
    const raw = await fs.readFile(ACTIVE_LAYER_FILE, "utf8");
    const data = JSON.parse(raw);
    res.json({ ...data, endpoint: COPERNICUS_ENDPOINT });
  } catch (err) {
    if (err.code === "ENOENT")
      return res.status(404).json({ error: "No active layer set" });
    console.error("[WMS] getActiveLayer error:", err);
    res.status(500).json({ error: "Failed to read active layer" });
  }
};

// Fetch & cache WMS tiles from Copernicus
exports.getWMSTile = async (req, res) => {
  try {
    const params = new URLSearchParams(req.query);
    const cacheKey = params.toString();

    const cached = tileCache.get(cacheKey);
    if (cached) {
      res.set("Content-Type", "image/png");
      res.set("Cache-Control", "public, max-age=604800"); 
      return res.send(cached);
    }

    let fetchPromise = inFlightRequests.get(cacheKey);
    if (!fetchPromise) {
      fetchPromise = (async () => {
        const wmsUrl = `${COPERNICUS_ENDPOINT}?${params.toString()}`;
        const response = await fetch(wmsUrl);

        if (!response.ok)
          throw new Error(`Failed to fetch WMS tile: ${response.status} ${response.statusText}`);

        const contentType = response.headers.get("content-type") || "";
        if (!contentType.includes("image")) {
          const text = await response.text();
          throw new Error(`Non-image WMS response: ${text.slice(0, 100)}...`);
        }

        const buffer = Buffer.from(await response.arrayBuffer());
        tileCache.set(cacheKey, buffer);
        return buffer;
      })();

      inFlightRequests.set(cacheKey, fetchPromise);
    }

    const buffer = await fetchPromise;
    res.set("Content-Type", "image/png");
    res.set("Cache-Control", "public, max-age=604800");
    res.send(buffer);
  } catch (err) {
    console.error("[WMS] getWMSTile error:", err);
    res.status(500).json({ error: "Failed to fetch WMS tile" });
  } finally {
    const params = new URLSearchParams(req.query);
    inFlightRequests.delete(params.toString());
  }
};
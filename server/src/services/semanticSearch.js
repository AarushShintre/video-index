import { exec } from "child_process";
import path from "path";
import util from "util";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();
const execPromise = util.promisify(exec);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PYTHON = process.env.PYTHON_PATH || "python3";

const PYTHON_SCRIPT = path.join(
  process.cwd(),
  "src/services/semanticSearchHelper.py"
);

export async function addVideoToIndex(videoPath) {
  const resultsDir = path.join(__dirname, "results");

  const payload = JSON.stringify({
    operation: "add_to_index",
    video_path: videoPath,
    results_dir: resultsDir,
  });

  const cmd = `"${PYTHON}" "${PYTHON_SCRIPT}" '${payload}'`;
  console.log("🔍 Running:", cmd);

  try {
    const { stdout } = await execPromise(cmd);
    console.log("✅ Python Output:", stdout.trim());

    const jsonMatch = stdout.trim().match(/\{.*\}/);
    if (!jsonMatch) throw new Error("No JSON returned");

    return JSON.parse(jsonMatch[0]);
  } catch (err) {
    console.error("❌ Python Error:", err);
    throw err;
  }
}

// Health check endpoint
router.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    index_loaded: modelsLoaded,
    index_size: modelsInfo.indexSize
  });
});

// List routes endpoint
router.get('/routes', (req, res) => {
  const routes = router.stack
    .filter(r => r.route)
    .map(r => ({
      path: r.route.path,
      methods: Object.keys(r.route.methods).filter(m => m !== '_all')
    }));
  
  res.json({ routes });
});

// Search endpoint
router.post('/search', async (req, res) => {
  try {
    // Try to load models if not already loaded
    if (!modelsLoaded) {
      console.log('Models not loaded, attempting to load...');
      const loaded = await loadModelsInfo();
      if (!loaded) {
        return res.status(503).json({ 
          error: 'Semantic search index not loaded',
          message: 'Please run video_clustering.py to generate embeddings and index'
        });
      }
    }

    const { video_path, video_id, k = 5 } = req.body;

    if (!video_path && !video_id) {
      return res.status(400).json({ error: 'Either video_path or video_id required' });
    }

    // If video_id provided, try to get path from database
    let finalVideoPath = video_path;
    if (video_id && !video_path) {
      try {
        const db = (await import('../config/database.js')).default;
        const video = await db.get('SELECT * FROM videos WHERE id = ?', [video_id]);
        if (video) {
          const uploadDir = process.env.UPLOAD_DIR || join(__dirname, '../../uploads');
          finalVideoPath = join(uploadDir, video.filepath);
        }
      } catch (dbError) {
        console.error('Database error:', dbError);
      }
    }

    // Handle URL paths - extract filename if it's a URL
    if (finalVideoPath && (finalVideoPath.includes('http://') || finalVideoPath.includes('https://'))) {
      const filename = finalVideoPath.split('/').pop();
      const projectRoot = resolve(__dirname, '../../../..');
      const uploadsDir = join(projectRoot, 'server', 'uploads');
      finalVideoPath = join(uploadsDir, filename);
    }

    if (!finalVideoPath || !existsSync(finalVideoPath)) {
      return res.status(404).json({ error: `Video file not found: ${finalVideoPath}` });
    }

    // Call Python helper for search
    const result = await callPythonHelper('search', {
      video_path: finalVideoPath,
      k: Math.min(k, modelsInfo.indexSize || 10)
    });

    res.json({
      results: result.results || [],
      count: result.count || 0
    });
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.stack 
    });
  }
});

// Add video endpoint
router.post('/add_video', async (req, res) => {
  try {
    if (!modelsLoaded) {
      const loaded = await loadModelsInfo();
      if (!loaded) {
        return res.status(503).json({ 
          error: 'Semantic search index not loaded',
          message: 'Please run video_clustering.py to generate embeddings and index'
        });
      }
    }

    const { video_path, video_id } = req.body;

    if (!video_path || !existsSync(video_path)) {
      return res.status(404).json({ error: `Video file not found: ${video_path}` });
    }

    // Call Python helper to extract embedding
    const result = await callPythonHelper('extract_embedding', {
      video_path: video_path,
      video_id: video_id
    });

    res.json({
      message: 'Video embedding extracted successfully',
      embedding_shape: result.embedding_shape,
      note: 'Full index rebuild required to search this video. New videos are searchable immediately as query videos.'
    });
  } catch (error) {
    console.error('Add video error:', error);
    res.status(500).json({ 
      error: error.message,
      traceback: error.stack 
    });
  }
});

// 404 handler
router.use((req, res) => {
  const routes = router.stack
    .filter(r => r.route)
    .map(r => ({
      path: r.route.path,
      methods: Object.keys(r.route.methods).filter(m => m !== '_all')
    }));

  res.status(404).json({
    error: 'Route not found',
    requested_path: req.path,
    requested_method: req.method,
    available_routes: routes
  });
});

// Initialize models on module load
loadModelsInfo().catch(err => {
  console.error('Failed to initialize models:', err);
});

export default router;


/**
 * Express service for semantic video search using FAISS and embeddings.
 * Calls Python scripts for embedding extraction and FAISS operations.
 */

import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join, resolve } from 'path';
import { existsSync, readFileSync } from 'fs';
import { promisify } from 'util';
import { exec } from 'child_process';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const router = express.Router();
router.use(cors());
router.use(express.json());

// Global variables for loaded models info
let modelsLoaded = false;
let modelsInfo = {
  indexSize: 0,
  embeddingsDir: null,
  normalizeEmbeddings: false,
  useCosine: false
};

/**
 * Find the results directory containing clustering results
 */
function findResultsDir() {
  const projectRoot = resolve(__dirname, '../../../..');
  const possiblePaths = [
    join(projectRoot, 'output'),
    join(__dirname, '../../output'),
    join(__dirname, '../../../output'),
    'output'
  ];

  for (const path of possiblePaths) {
    const resolvedPath = resolve(path);
    const resultsFile = join(resolvedPath, 'clustering_results.npz');
    if (existsSync(resultsFile)) {
      return resolvedPath;
    }
  }

  return null;
}

/**
 * Find Python executable (python3 or python)
 */
async function findPythonCommand() {
  try {
    await execAsync('python3 --version');
    return 'python3';
  } catch {
    try {
      await execAsync('python --version');
      return 'python';
    } catch {
      return 'python'; // Default fallback
    }
  }
}

/**
 * Load semantic search models info
 */
async function loadModelsInfo() {
  const resultsDir = findResultsDir();
  if (!resultsDir) {
    console.error('⚠️  Results directory not found. Semantic search will not be available.');
    console.error('   Make sure you have run video_clustering.py to generate embeddings.');
    return false;
  }

  try {
    const pythonCmd = await findPythonCommand();
    const pythonScript = join(__dirname, 'semanticSearchHelper.py');
    
    if (!existsSync(pythonScript)) {
      console.error(`❌ Python helper script not found: ${pythonScript}`);
      return false;
    }
    
    console.log(`Loading semantic search models from: ${resultsDir}`);
    // On Windows, use proper quoting; on Unix, paths should work as-is
    const command = `"${pythonCmd}" "${pythonScript}" --operation info --results-dir "${resultsDir}"`;
    
    const { stdout, stderr } = await execAsync(command);
    
    if (stderr && !stdout) {
      console.error('Python script error:', stderr);
      return false;
    }
    
    const info = JSON.parse(stdout);
    
    if (info.error) {
      console.error('Failed to load models:', info.error);
      return false;
    }
    
    modelsInfo = {
      indexSize: info.index_size || 0,
      embeddingsDir: resultsDir,
      normalizeEmbeddings: info.normalize_embeddings || false,
      useCosine: info.use_cosine || false
    };
    modelsLoaded = true;
    console.log(`✅ Loaded semantic search models from: ${resultsDir}`);
    console.log(`   Index size: ${modelsInfo.indexSize} vectors`);
    return true;
  } catch (error) {
    console.error('❌ Failed to load models info:', error.message);
    if (error.stdout) console.error('   stdout:', error.stdout);
    if (error.stderr) console.error('   stderr:', error.stderr);
    return false;
  }
}

/**
 * Call Python helper script for semantic search
 */
async function callPythonHelper(operation, data) {
  const resultsDir = modelsInfo.embeddingsDir || findResultsDir();
  if (!resultsDir) {
    throw new Error('Results directory not found');
  }

  const pythonCmd = await findPythonCommand();
  const pythonScript = join(__dirname, 'semanticSearchHelper.py');
  const args = [
    pythonScript,
    '--operation', operation,
    '--results-dir', resultsDir,
    '--data', JSON.stringify(data)
  ];

  return new Promise((resolve, reject) => {
    const python = spawn(pythonCmd, args, {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      if (code !== 0) {
        const errorMsg = stderr || stdout || 'Unknown error';
        console.error(`Python script failed (code ${code}):`, errorMsg);
        reject(new Error(`Python script failed: ${errorMsg}`));
      } else {
        try {
          if (!stdout.trim()) {
            reject(new Error('Python script returned empty output'));
            return;
          }
          const result = JSON.parse(stdout);
          if (result.error) {
            reject(new Error(result.error));
          } else {
            resolve(result);
          }
        } catch (e) {
          console.error('Failed to parse Python output:', stdout);
          console.error('Parse error:', e.message);
          reject(new Error(`Failed to parse Python output: ${e.message}`));
        }
      }
    });

    python.on('error', (error) => {
      console.error('Failed to spawn Python process:', error);
      reject(new Error(`Failed to spawn Python process: ${error.message}`));
    });
  });
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


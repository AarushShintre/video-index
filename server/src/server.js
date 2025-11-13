import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import videoRoutes from './routes/videos.js';
import semanticSearchRoutes from './services/semanticSearch.js';
import { initDatabase, resetDatabase } from './config/database.js';
import { syncDatabaseWithUploads } from './services/videoSync.js';
import { runClusteringPipeline, clusteringResultsExist } from './services/clusteringService.js';
import { watchUploadsFolder } from './services/fileWatcher.js';
import { loadModelsInfo } from './services/semanticSearch.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const UPLOAD_DIR = process.env.UPLOAD_DIR 
  ? path.resolve(process.env.UPLOAD_DIR)
  : path.resolve(__dirname, '../../uploads');
const OUTPUT_DIR = process.env.OUTPUT_DIR || 'output';

// Global flag to prevent concurrent rebuilds
let isRebuilding = false;
let rebuildQueue = false;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));
app.use('/uploads', express.static(UPLOAD_DIR));



app.use('/api/videos', videoRoutes);

// ✅ both semantic routes properly mounted
app.use('/api/semantic-search', semanticSearchRoutes);


app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

/**
 * Rebuild clustering and semantic search index
 */
async function rebuildClustering() {
  if (isRebuilding) {
    console.log('Rebuild already in progress, queueing rebuild...');
    rebuildQueue = true;
    return;
  }

  isRebuilding = true;
  rebuildQueue = false;

  try {
    console.log('\nStarting rebuild process...');

    // Step 1: Sync database with uploads folder
    console.log('Step 1: Syncing database with uploads folder...');
    await syncDatabaseWithUploads(UPLOAD_DIR);

    // Step 2: Run clustering pipeline
    console.log('Step 2: Running clustering pipeline...');
    await runClusteringPipeline(UPLOAD_DIR, OUTPUT_DIR, {
      maxVideos: 100,
      normalize: true,
      useCosine: true,
      skipFrames: 4,
      pcaComponents: 128,
      nClusters: 10
    });

    // Step 3: Reload semantic search models
    console.log('Step 3: Reloading semantic search models...');
    try {
      await loadModelsInfo();
      console.log('Semantic search models reloaded');
    } catch (error) {
      console.warn('Warning: Could not reload semantic search models:', error.message);
    }

    console.log('Rebuild process completed successfully\n');

    // Check if another rebuild was requested during this rebuild
    if (rebuildQueue) {
      console.log('Processing queued rebuild request...');
      setTimeout(() => rebuildClustering(), 1000);
    }
  } catch (error) {
    console.error('Error during rebuild:', error);
  } finally {
    isRebuilding = false;
  }
}

/**
 * Initialize server: reset DB, sync, cluster, and start watching
 */
async function initializeServer() {
  try {
    console.log('\nInitializing server...\n');

    // Step 1: Initialize database
    console.log('Step 1: Initializing database...');
    await initDatabase();

    // Step 2: Reset database (clear old records)
    console.log('Step 2: Resetting database...');
    await resetDatabase();

    // Step 3: Sync database with uploads folder
    console.log('Step 3: Syncing database with uploads folder...');
    const syncResult = await syncDatabaseWithUploads(UPLOAD_DIR);
    
    if (syncResult.total === 0) {
      console.log('No videos found in uploads folder. Clustering will be skipped.');
      console.log('   Upload videos to start clustering and semantic search.\n');
    } else {
      // Step 4: Run clustering pipeline
      console.log('Step 4: Running clustering pipeline...');
      try {
        await runClusteringPipeline(UPLOAD_DIR, OUTPUT_DIR, {
          maxVideos: 100,
          normalize: true,
          useCosine: true,
          skipFrames: 4,
          pcaComponents: 128,
          nClusters: 10
        });
      } catch (error) {
        console.error('Error during clustering:', error.message);
        console.log('Continuing without clustering results...');
      }

      // Step 5: Load semantic search models
      console.log('Step 5: Loading semantic search models...');
      try {
        await loadModelsInfo();
        console.log('Semantic search models loaded');
      } catch (error) {
        console.warn('Warning: Could not load semantic search models:', error.message);
        console.log('   Semantic search will not be available until clustering completes.');
      }
    }

    // Step 6: Start file watcher
    console.log('Step 6: Starting file watcher...');
    watchUploadsFolder(UPLOAD_DIR, rebuildClustering);

    console.log('\nServer initialization complete!\n');
  } catch (error) {
    console.error('Error during server initialization:', error);
    throw error;
  }
}

// Export rebuild function for use in controllers
export { rebuildClustering };

// Initialize server and start listening
initializeServer().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`Uploads directory: ${UPLOAD_DIR}`);
    console.log(`Output directory: ${OUTPUT_DIR}\n`);
  });
}).catch(error => {
  console.error('Failed to initialize server:', error);
  process.exit(1);
});

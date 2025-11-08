import axios from 'axios';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SEMANTIC_SEARCH_URL = process.env.SEMANTIC_SEARCH_URL || 'http://localhost:5001';

/**
 * Find similar videos using semantic search
 */
export const findSimilarVideos = async (req, res) => {
  try {
    const videoId = req.params.id || req.body.video_id;
    const { video_path, k = 5 } = req.body;
    
    // Get video from database to get filepath
    const db = (await import('../config/database.js')).default;
    const video = await db.get('SELECT * FROM videos WHERE id = ?', [videoId]);
    
    if (!video) {
      return res.status(404).json({ error: 'Video not found' });
    }

    // Construct full path to video file
    const uploadDir = process.env.UPLOAD_DIR || path.join(__dirname, '../../uploads');
    const fullVideoPath = path.join(uploadDir, video.filepath);

    // Call Python semantic search service
    const response = await axios.post(`${SEMANTIC_SEARCH_URL}/search`, {
      video_id: videoId,
      video_path: fullVideoPath,
      k
    });

    res.json(response.data);
  } catch (error) {
    console.error('Semantic search error:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ 
        error: 'Semantic search service unavailable',
        message: 'Make sure the Python semantic search service is running on port 5001'
      });
    }
    
    res.status(500).json({ 
      error: error.message,
      details: error.response?.data 
    });
  }
};

/**
 * Check if semantic search service is available
 */
export const checkSemanticSearchHealth = async (req, res) => {
  try {
    const response = await axios.get(`${SEMANTIC_SEARCH_URL}/health`);
    res.json(response.data);
  } catch (error) {
    res.status(503).json({ 
      status: 'unavailable',
      error: error.message 
    });
  }
};


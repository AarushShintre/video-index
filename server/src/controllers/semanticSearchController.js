import axios from 'axios';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Use local semantic search service (same Express app)
const SEMANTIC_SEARCH_URL = process.env.SEMANTIC_SEARCH_URL || `http://localhost:${process.env.PORT || 5000}/api/semantic-search`;

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

    // Call local semantic search service
    const response = await axios.post(`${SEMANTIC_SEARCH_URL}/search`, {
      video_id: videoId,
      video_path: fullVideoPath,
      k
    });

    res.json(response.data);
  } catch (error) {
    console.error('Semantic search error:', error.message);
    console.error('Full error:', error);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ 
        error: 'Semantic search service unavailable',
        message: 'Semantic search service is not responding'
      });
    }
    
    // If the semantic search service returned an error, pass it through
    if (error.response?.data) {
      return res.status(error.response.status || 500).json(error.response.data);
    }
    
    res.status(500).json({ 
      error: error.message,
      details: error.stack 
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

/**
 * Process a newly uploaded video for semantic search
 * Extracts embeddings and finds similar videos
 * @param {string} videoId - The video ID in the database
 * @param {string} videoPath - Full path to the video file
 * @returns {Object} Object containing similar videos or null if service unavailable
 */
export const processVideoForSemanticSearch = async (videoId, videoPath) => {
  try {
    // First, check if semantic search service is available
    try {
      await axios.get(`${SEMANTIC_SEARCH_URL}/health`, { timeout: 2000 });
    } catch (healthError) {
      console.warn('Semantic search service not available:', healthError.message);
      return null;
    }

    // Extract embedding and add to index (if supported)
    try {
      await axios.post(`${SEMANTIC_SEARCH_URL}/add_video`, {
        video_id: videoId,
        video_path: videoPath
      }, { timeout: 30000 }); // 30 second timeout for embedding extraction
    } catch (addError) {
      console.warn('Failed to add video to semantic search index:', addError.message);
      // Continue to try searching even if adding fails
    }

    // Search for similar videos
    try {
      const searchResponse = await axios.post(`${SEMANTIC_SEARCH_URL}/search`, {
        video_id: videoId,
        video_path: videoPath,
        k: 5
      }, { timeout: 30000 }); // 30 second timeout for search

      return {
        success: true,
        similarVideos: searchResponse.data.results || [],
        count: searchResponse.data.count || 0
      };
    } catch (searchError) {
      console.warn('Failed to search for similar videos:', searchError.message);
      return {
        success: false,
        error: 'Search failed',
        message: searchError.message
      };
    }

  } catch (error) {
    console.error('Error processing video for semantic search:', error.message);
    return null;
  }
};

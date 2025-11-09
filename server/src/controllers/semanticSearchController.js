import fs from "fs";
import path from "path";
import { addVideoToIndex } from "../services/semanticSearch.js";

export const checkSemanticSearchHealth = async (req, res) => {
  return res.json({ status: "ok", message: "Semantic search service running" });
};

export const updateSemanticIndex = async (req, res) => {
  try {
    const { video_path } = req.body;

    if (!video_path) {
      return res.status(400).json({ error: "video_path missing" });
    }

    const fullPath = path.join(process.env.UPLOAD_DIR, path.basename(video_path));

    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ error: "File not found on server" });
    }

    await addVideoToIndex(fullPath);

    return res.json({ success: true, message: "Added to semantic index" });

  } catch (err) {
    console.error("❌ Index update failed:", err);
    return res.status(500).json({ error: "Could not add to index" });
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

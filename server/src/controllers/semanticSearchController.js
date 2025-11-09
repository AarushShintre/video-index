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

export const findSimilarVideos = async (req, res) => {
  res.json({
    success: false,
    message: "Semantic similarity function not implemented yet"
  });
};

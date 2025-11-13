import fs from "fs";

export const updateIndex = async (req, res) => {
  try {
    const { video_path } = req.body;
    if (!video_path || !fs.existsSync(video_path)) {
      return res.status(400).json({ error: "Invalid video path" });
    }

    console.log("✅ Indexing video:", video_path);

    // TODO: Add ML embedding later
    res.json({ status: "ok", indexed: video_path });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

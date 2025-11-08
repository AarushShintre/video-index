import db from '../config/database.js';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import path from 'path';

export const getAllVideos = async (req, res) => {
  try {
    const { search, sort = 'newest' } = req.query;
    
    let query = 'SELECT * FROM videos';
    let params = [];

    if (search) {
      query += ' WHERE title LIKE ? OR description LIKE ? OR transcription LIKE ?';
      const searchTerm = `%${search}%`;
      params = [searchTerm, searchTerm, searchTerm];
    }

    switch (sort) {
      case 'oldest':
        query += ' ORDER BY created_at ASC';
        break;
      case 'mostViewed':
        query += ' ORDER BY views DESC';
        break;
      case 'title':
        query += ' ORDER BY title ASC';
        break;
      default:
        query += ' ORDER BY created_at DESC';
    }

    const videos = await db.all(query, params);
    res.json(videos);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

export const getVideoById = async (req, res) => {
  try {
    const video = await db.get('SELECT * FROM videos WHERE id = ?', [req.params.id]);
    if (!video) {
      return res.status(404).json({ error: 'Video not found' });
    }
    
    await db.run('UPDATE videos SET views = views + 1 WHERE id = ?', [req.params.id]);
    video.views += 1;
    
    res.json(video);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

export const uploadVideo = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file uploaded' });
    }

    const videoId = uuidv4();
    const { title, description, transcription, tags } = req.body;

    await db.run(
      `INSERT INTO videos (id, title, description, filename, filepath, size, type, transcription, tags, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        videoId,
        title || req.file.originalname,
        description || '',
        req.file.originalname,
        req.file.filename,
        req.file.size,
        req.file.mimetype,
        transcription || '',
        tags || '',
        Date.now()
      ]
    );

    const video = await db.get('SELECT * FROM videos WHERE id = ?', [videoId]);
    res.status(201).json(video);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

export const updateVideo = async (req, res) => {
  try {
    const { title, description, transcription, tags } = req.body;
    
    await db.run(
      'UPDATE videos SET title = ?, description = ?, transcription = ?, tags = ? WHERE id = ?',
      [title, description, transcription, tags, req.params.id]
    );

    const video = await db.get('SELECT * FROM videos WHERE id = ?', [req.params.id]);
    res.json(video);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

export const deleteVideo = async (req, res) => {
  try {
    const video = await db.get('SELECT * FROM videos WHERE id = ?', [req.params.id]);
    if (!video) {
      return res.status(404).json({ error: 'Video not found' });
    }

    const filePath = path.join(process.env.UPLOAD_DIR, video.filepath);
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }

    await db.run('DELETE FROM videos WHERE id = ?', [req.params.id]);
    res.json({ message: 'Video deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
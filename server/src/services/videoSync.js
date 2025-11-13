import db from '../config/database.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Get all video files from uploads directory
 */
export function getVideoFilesFromUploads(uploadsDir) {
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
    return [];
  }

  const videoExtensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.MOV', '.MP4', '.AVI', '.WEBM', '.MKV'];
  const files = fs.readdirSync(uploadsDir);
  
  return files
    .filter(file => {
      const ext = path.extname(file).toLowerCase();
      return videoExtensions.includes(ext);
    })
    .map(file => ({
      filename: file,
      filepath: file,
      fullPath: path.join(uploadsDir, file)
    }));
}

/**
 * Sync SQLite database with uploads folder
 * Adds videos that exist in uploads but not in database
 * Removes videos from database that don't exist in uploads
 */
export async function syncDatabaseWithUploads(uploadsDir) {
  try {
    const videoFiles = getVideoFilesFromUploads(uploadsDir);
    const dbVideos = await db.all('SELECT * FROM videos');
    
    // Create a map of filepaths for quick lookup
    const dbVideoMap = new Map();
    dbVideos.forEach(video => {
      dbVideoMap.set(video.filepath, video);
    });

    // Create a set of existing filepaths
    const existingFiles = new Set(videoFiles.map(v => v.filepath));

    // Remove videos from database that don't exist in uploads folder
    const videosToRemove = [];
    for (const video of dbVideos) {
      if (!existingFiles.has(video.filepath)) {
        videosToRemove.push(video.id);
      }
    }

    if (videosToRemove.length > 0) {
      console.log(`Removing ${videosToRemove.length} videos from database that no longer exist in uploads folder`);
      const placeholders = videosToRemove.map(() => '?').join(',');
      await db.run(`DELETE FROM videos WHERE id IN (${placeholders})`, videosToRemove);
    }

    // Add videos that exist in uploads but not in database
    const videosToAdd = [];
    for (const videoFile of videoFiles) {
      if (!dbVideoMap.has(videoFile.filepath)) {
        try {
          const stats = fs.statSync(videoFile.fullPath);
          const videoId = uuidv4();
          const title = path.basename(videoFile.filename, path.extname(videoFile.filename));
          
          videosToAdd.push({
            id: videoId,
            title: title,
            description: '',
            filename: videoFile.filename,
            filepath: videoFile.filepath,
            size: stats.size,
            type: 'video/mp4', // Default type, could be improved by checking mime type
            created_at: stats.mtimeMs || Date.now()
          });
        } catch (error) {
          console.error(`Error processing video file ${videoFile.filename}:`, error.message);
        }
      }
    }

    if (videosToAdd.length > 0) {
      console.log(`Adding ${videosToAdd.length} new videos to database`);
      for (const video of videosToAdd) {
        await db.run(
          `INSERT INTO videos (id, title, description, filename, filepath, size, type, transcription, tags, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          [
            video.id,
            video.title,
            video.description,
            video.filename,
            video.filepath,
            video.size,
            video.type,
            '',
            '',
            video.created_at
          ]
        );
      }
    }

    const finalCount = await db.get('SELECT COUNT(*) as count FROM videos');
    console.log(`Database sync complete. Total videos in database: ${finalCount.count}`);
    
    return {
      added: videosToAdd.length,
      removed: videosToRemove.length,
      total: finalCount.count
    };
  } catch (error) {
    console.error('Error syncing database with uploads:', error);
    throw error;
  }
}



import chokidar from 'chokidar';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let watcher = null;
let rebuildTimeout = null;
const REBUILD_DELAY = 5000; // Wait 5 seconds after last change before rebuilding

/**
 * Watch uploads directory for changes and trigger rebuild
 */
export function watchUploadsFolder(uploadsDir, onChange) {
  if (watcher) {
    watcher.close();
  }

  console.log(`Watching uploads folder: ${uploadsDir}`);

  watcher = chokidar.watch(uploadsDir, {
    ignored: /(^|[\/\\])\../, // ignore dotfiles
    persistent: true,
    ignoreInitial: true, // Don't trigger on initial scan
    awaitWriteFinish: {
      stabilityThreshold: 2000,
      pollInterval: 100
    }
  });

  const videoExtensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.MOV', '.MP4', '.AVI', '.WEBM', '.MKV'];

  const handleChange = (filePath) => {
    const ext = path.extname(filePath).toLowerCase();
    if (!videoExtensions.includes(ext)) {
      return; // Not a video file
    }

    console.log(`Detected change in uploads folder: ${path.basename(filePath)}`);

    // Clear existing timeout
    if (rebuildTimeout) {
      clearTimeout(rebuildTimeout);
    }

    // Set new timeout to debounce rebuilds
    rebuildTimeout = setTimeout(async () => {
      console.log('Triggering rebuild after file change...');
      try {
        await onChange();
      } catch (error) {
        console.error('Error during rebuild after file change:', error);
      }
    }, REBUILD_DELAY);
  };

  watcher
    .on('add', handleChange)
    .on('unlink', handleChange)
    .on('change', handleChange)
    .on('error', error => {
      console.error('File watcher error:', error);
    })
    .on('ready', () => {
      console.log('File watcher ready');
    });

  return watcher;
}

/**
 * Stop watching uploads folder
 */
export function stopWatching() {
  if (watcher) {
    watcher.close();
    watcher = null;
  }
  if (rebuildTimeout) {
    clearTimeout(rebuildTimeout);
    rebuildTimeout = null;
  }
}


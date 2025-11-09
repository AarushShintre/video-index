import express from 'express';
import { upload } from '../middleware/upload.js';
import {
  getAllVideos,
  getVideoById,
  uploadVideo,
  updateVideo,
  deleteVideo
} from '../controllers/videoController.js';
import {
  findSimilarVideos,
  checkSemanticSearchHealth
} from '../controllers/semanticSearchController.js';

const router = express.Router();

router.get('/', getAllVideos);
router.get('/:id', getVideoById);
router.post('/', upload.single('video'), uploadVideo);
router.put('/:id', updateVideo);
router.delete('/:id', deleteVideo);

// Semantic search routes
router.post('/:id/similar', findSimilarVideos);
router.get('/semantic-search/health', checkSemanticSearchHealth);



export default router;
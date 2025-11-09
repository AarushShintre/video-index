import axios from 'axios';

// ✅ Base URL (no change needed)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const videoAPI = {
  getAllVideos: (params) => api.get('/videos', { params }),
  getVideoById: (id) => api.get(`/videos/${id}`),
  uploadVideo: (formData, onProgress) => 
    api.post('/videos', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onProgress
    }),
  updateVideo: (id, data) => api.put(`/videos/${id}`, data),
  deleteVideo: (id) => api.delete(`/videos/${id}`),

  // ✅ Fixed path — semantic search lives under /api/semantic-search
  findSimilarVideos: (videoId, videoPath, k = 5) =>
    api.post(`/semantic-search/${videoId}/similar`, {
      video_id: videoId,
      video_path: videoPath,
      k
    }),

  // ✅ Correct health endpoint
  checkSemanticSearchHealth: () =>
    api.get(`/semantic-search/health`),

  // ✅ NEW — endpoint to update index after upload
  updateSemanticIndex: (videoPath) =>
    api.post(`/semantic-search/update-index`, {
      video_path: videoPath
    })
};

export default api;

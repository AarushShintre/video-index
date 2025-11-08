import React, { useState, useEffect, useRef } from 'react';
import { Upload, Play, X, Search, Grid3x3, List, Trash2, Edit2, Save, Film, Eye, Calendar, Sparkles, Loader2 } from 'lucide-react';
import { videoAPI } from './services/api';

function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState('grid');
  const [sortBy, setSortBy] = useState('newest');
  const [editing, setEditing] = useState(false);
  const [editData, setEditData] = useState({});
  const [similarVideos, setSimilarVideos] = useState([]);
  const [loadingSimilar, setLoadingSimilar] = useState(false);
  const [semanticSearchAvailable, setSemanticSearchAvailable] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadVideos();
    checkSemanticSearch();
  }, [searchTerm, sortBy]);

  const checkSemanticSearch = async () => {
    try {
      const response = await videoAPI.checkSemanticSearchHealth();
      setSemanticSearchAvailable(response.data.status === 'ok');
    } catch (error) {
      setSemanticSearchAvailable(false);
    }
  };

  const loadSimilarVideos = async (videoId, videoPath) => {
    if (!semanticSearchAvailable) return;
    
    setLoadingSimilar(true);
    try {
      const response = await videoAPI.findSimilarVideos(videoId, videoPath, 5);
      setSimilarVideos(response.data.results || []);
    } catch (error) {
      console.error('Error loading similar videos:', error);
      setSimilarVideos([]);
    } finally {
      setLoadingSimilar(false);
    }
  };

  const loadVideos = async () => {
    try {
      const response = await videoAPI.getAllVideos({ search: searchTerm, sort: sortBy });
      setVideos(response.data);
    } catch (error) {
      console.error('Error loading videos:', error);
    }
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    setUploading(true);

    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append('video', file);
        formData.append('title', file.name.replace(/\.[^/.]+$/, ''));

        await videoAPI.uploadVideo(formData, (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        });
      } catch (error) {
        console.error('Upload error:', error);
        alert(`Failed to upload ${file.name}`);
      }
    }

    setUploading(false);
    setUploadProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
    loadVideos();
  };

  const handleVideoClick = async (videoId) => {
    try {
      const response = await videoAPI.getVideoById(videoId);
      const video = response.data;
      setSelectedVideo(video);
      setEditData(video);
      setEditing(false);
      
      // Load similar videos if semantic search is available
      if (semanticSearchAvailable) {
        // Pass both video_id and filepath - the service will handle path resolution
        loadSimilarVideos(videoId, video.filepath);
      }
    } catch (error) {
      console.error('Error loading video:', error);
    }
  };

  const handleSaveEdit = async () => {
    try {
      await videoAPI.updateVideo(selectedVideo.id, editData);
      setSelectedVideo(editData);
      setEditing(false);
      loadVideos();
    } catch (error) {
      console.error('Error updating video:', error);
    }
  };

  const handleDelete = async (videoId) => {
    if (!confirm('Delete this video?')) return;
    
    try {
      await videoAPI.deleteVideo(videoId);
      if (selectedVideo?.id === videoId) {
        setSelectedVideo(null);
      }
      loadVideos();
    } catch (error) {
      console.error('Error deleting video:', error);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDuration = (seconds) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return `${Math.floor(diffDays / 365)} years ago`;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-screen-2xl mx-auto px-6 py-3">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-fit">
              <div className="bg-red-600 p-2 rounded-lg">
                <Film className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-xl font-bold text-gray-900">VideoIndex</h1>
            </div>

            <div className="flex-1 max-w-2xl">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search videos by title, description, or content..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-full pl-12 pr-4 py-2.5 focus:outline-none focus:border-red-500 focus:bg-white"
                />
              </div>
            </div>

            <label className="cursor-pointer">
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                multiple
                onChange={handleFileUpload}
                className="hidden"
                disabled={uploading}
              />
              <div className="flex items-center gap-2 bg-red-600 hover:bg-red-700 px-4 py-2.5 rounded-full text-white font-medium">
                <Upload className="w-5 h-5" />
                <span>{uploading ? `${uploadProgress}%` : 'Upload'}</span>
              </div>
            </label>
          </div>
        </div>
      </header>

      <div className="max-w-screen-2xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex bg-white rounded-lg p-1 border border-gray-200">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded ${viewMode === 'grid' ? 'bg-gray-200' : 'hover:bg-gray-100'}`}
              >
                <Grid3x3 className="w-5 h-5" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded ${viewMode === 'list' ? 'bg-gray-200' : 'hover:bg-gray-100'}`}
              >
                <List className="w-5 h-5" />
              </button>
            </div>
            <span className="text-gray-600 font-medium">{videos.length} videos</span>
          </div>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-white border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:border-red-500"
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
            <option value="mostViewed">Most viewed</option>
            <option value="title">Title (A-Z)</option>
          </select>
        </div>
      </div>

      <main className="max-w-screen-2xl mx-auto px-6 pb-12">
        {videos.length === 0 ? (
          <div className="text-center py-20">
            <Film className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">No videos yet</h2>
            <p className="text-gray-600">Upload your first video to get started</p>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {videos.map((video) => (
              <div
                key={video.id}
                className="bg-white rounded-xl overflow-hidden shadow-md hover:shadow-xl transition cursor-pointer group"
              >
                <div
                  onClick={() => handleVideoClick(video.id)}
                  className="relative aspect-video bg-gray-900"
                >
                  <video src={`http://localhost:5000/uploads/${video.filepath}`} className="w-full h-full object-cover" />
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 flex items-center justify-center transition">
                    <div className="transform scale-0 group-hover:scale-100 transition">
                      <div className="bg-red-600 rounded-full p-4">
                        <Play className="w-8 h-8 text-white fill-white" />
                      </div>
                    </div>
                  </div>
                  {video.duration > 0 && (
                    <div className="absolute bottom-2 right-2 bg-black bg-opacity-80 px-2 py-1 rounded text-xs text-white">
                      {formatDuration(video.duration)}
                    </div>
                  )}
                </div>
                
                <div className="p-4">
                  <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2 group-hover:text-red-600">
                    {video.title}
                  </h3>
                  <div className="flex items-center gap-3 text-sm text-gray-600 mb-3">
                    <div className="flex items-center gap-1">
                      <Eye className="w-4 h-4" />
                      <span>{video.views}</span>
                    </div>
                    <span>•</span>
                    <span>{formatDate(video.created_at)}</span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(video.id);
                    }}
                    className="w-full flex items-center justify-center gap-2 bg-red-50 hover:bg-red-100 text-red-600 px-3 py-2 rounded transition text-sm"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            {videos.map((video) => (
              <div
                key={video.id}
                onClick={() => handleVideoClick(video.id)}
                className="bg-white rounded-xl shadow-md hover:shadow-xl transition cursor-pointer flex gap-4 p-4"
              >
                <div className="relative w-64 aspect-video bg-gray-900 rounded-lg overflow-hidden">
                  <video src={`http://localhost:5000/uploads/${video.filepath}`} className="w-full h-full object-cover" />
                </div>
                
                <div className="flex-1">
                  <h3 className="font-semibold text-lg text-gray-900 mb-2">{video.title}</h3>
                  <div className="flex items-center gap-3 text-sm text-gray-600">
                    <span>{video.views} views</span>
                    <span>•</span>
                    <span>{formatDate(video.created_at)}</span>
                    <span>•</span>
                    <span>{formatBytes(video.size)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {selectedVideo && (
        <div className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-white rounded-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
            <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between z-10">
              <h2 className="text-xl font-bold truncate flex-1">
                {editing ? (
                  <input
                    type="text"
                    value={editData.title}
                    onChange={(e) => setEditData({...editData, title: e.target.value})}
                    className="w-full border-b-2 border-red-500 focus:outline-none"
                  />
                ) : (
                  selectedVideo.title
                )}
              </h2>
              <div className="flex gap-2 ml-4">
                {editing ? (
                  <button onClick={handleSaveEdit} className="p-2 bg-green-500 text-white rounded-full hover:bg-green-600">
                    <Save className="w-5 h-5" />
                  </button>
                ) : (
                  <button onClick={() => setEditing(true)} className="p-2 hover:bg-gray-100 rounded-full">
                    <Edit2 className="w-5 h-5" />
                  </button>
                )}
                <button onClick={() => setSelectedVideo(null)} className="p-2 hover:bg-gray-100 rounded-full">
                  <X className="w-6 h-6" />
                </button>
              </div>
            </div>

            <div className="p-6">
              <video
                src={`http://localhost:5000/uploads/${selectedVideo.filepath}`}
                controls
                autoPlay
                className="w-full aspect-video bg-black rounded-lg mb-6"
              />

              <div className="space-y-6">
                <div className="flex items-center gap-6 text-sm text-gray-600 pb-4 border-b">
                  <div className="flex items-center gap-2">
                    <Eye className="w-5 h-5" />
                    <span>{selectedVideo.views} views</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Calendar className="w-5 h-5" />
                    <span>{formatDate(selectedVideo.created_at)}</span>
                  </div>
                  <span>{formatBytes(selectedVideo.size)}</span>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-900 mb-2">Description</label>
                  {editing ? (
                    <textarea
                      value={editData.description || ''}
                      onChange={(e) => setEditData({...editData, description: e.target.value})}
                      className="w-full border border-gray-300 rounded-lg p-4 focus:outline-none focus:border-red-500 min-h-24"
                    />
                  ) : (
                    <div className="bg-gray-50 rounded-lg p-4 border">
                      {selectedVideo.description || 'No description'}
                    </div>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-900 mb-2">
                    Transcription / Content (for ML Search)
                  </label>
                  {editing ? (
                    <textarea
                      value={editData.transcription || ''}
                      onChange={(e) => setEditData({...editData, transcription: e.target.value})}
                      placeholder="Add transcription or content here..."
                      className="w-full border border-gray-300 rounded-lg p-4 focus:outline-none focus:border-red-500 min-h-32"
                    />
                  ) : (
                    <div className="bg-gray-50 rounded-lg p-4 border min-h-32">
                      {selectedVideo.transcription || 'No transcription yet'}
                    </div>
                  )}
                  <p className="text-xs text-gray-500 mt-2">
                    💡 In production: Use AWS Transcribe to auto-generate this
                  </p>
                </div>

                {semanticSearchAvailable && (
                  <div>
                    <label className="block text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-purple-600" />
                      Similar Videos (Semantic Search)
                    </label>
                    {loadingSimilar ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="w-6 h-6 animate-spin text-purple-600" />
                        <span className="ml-2 text-gray-600">Finding similar videos...</span>
                      </div>
                    ) : similarVideos.length > 0 ? (
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        {similarVideos.map((result, idx) => {
                          // Extract video filename from path
                          const filename = result.video_path.split(/[/\\]/).pop();
                          const similarity = Math.round(result.similarity * 100);
                          
                          return (
                            <div
                              key={idx}
                              className="bg-gray-50 rounded-lg p-3 border border-gray-200 hover:border-purple-400 transition cursor-pointer"
                              onClick={() => {
                                // Find video by filename in our videos list
                                const video = videos.find(v => v.filepath === filename || v.filename === filename);
                                if (video) {
                                  handleVideoClick(video.id);
                                }
                              }}
                            >
                              <div className="aspect-video bg-gray-900 rounded mb-2 overflow-hidden">
                                <video 
                                  src={`http://localhost:5000/uploads/${filename}`}
                                  className="w-full h-full object-cover"
                                  muted
                                />
                              </div>
                              <div className="text-xs text-gray-600">
                                <div className="font-medium text-gray-900 truncate">{filename}</div>
                                <div className="text-purple-600 mt-1">{similarity}% similar</div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="bg-gray-50 rounded-lg p-4 border text-center text-gray-500 text-sm">
                        No similar videos found. Make sure videos are indexed for semantic search.
                      </div>
                    )}
                    <p className="text-xs text-gray-500 mt-2">
                      💡 Similar videos are found using visual content analysis (ResNet-50 embeddings)
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
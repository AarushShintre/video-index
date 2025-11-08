# VideoHub - Video Indexing Platform

A modern, full-stack video management platform that allows you to upload, organize, search, and manage your video library with AI-powered content search capabilities.

![VideoHub](https://img.shields.io/badge/VideoHub-v1.0.0-red)
![React](https://img.shields.io/badge/React-19.1.1-blue)
![Express](https://img.shields.io/badge/Express-4.18.2-green)

## ✨ Features

- **📤 Video Upload** - Upload multiple videos with progress tracking
- **🔍 Smart Search** - Search videos by title, description, or transcription content
- **📊 Video Management** - View, edit, and delete videos with detailed metadata
- **🎨 Modern UI** - Beautiful, responsive interface with smooth animations
- **📱 Multiple View Modes** - Grid and list view options
- **🔎 ML-Ready Search** - Transcription-based content search (ready for AWS Transcribe integration)
- **📈 View Tracking** - Track video views and engagement
- **📅 Smart Date Formatting** - Human-readable relative dates

## 🛠️ Tech Stack

### Frontend
- **React 19** - Modern React with hooks
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icon library
- **Axios** - HTTP client for API requests

### Backend
- **Express.js** - Web framework
- **SQLite** - Lightweight database
- **Multer** - File upload handling
- **CORS** - Cross-origin resource sharing

## 📋 Prerequisites

- Node.js (v18 or higher)
- npm or yarn

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd video-index
```

### 2. Install Dependencies

#### Backend
```bash
cd server
npm install
```

#### Frontend
```bash
cd ../client
npm install
```

### 3. Configure Environment Variables

Create a `.env` file in the `server` directory:

```env
PORT=5000
UPLOAD_DIR=./uploads
```

Create a `.env` file in the `client` directory (optional):

```env
VITE_API_URL=http://localhost:5000/api
```

### 4. Run the Application

#### Start the Backend Server

```bash
cd server
npm run dev
```

The server will start on `http://localhost:5000`

#### Start the Frontend Development Server

In a new terminal:

```bash
cd client
npm run dev
```

The client will start on `http://localhost:5173` (or another port if 5173 is busy)

## 📁 Project Structure

```
video-index/
├── client/                 # React frontend application
│   ├── src/
│   │   ├── App.jsx        # Main application component
│   │   ├── services/
│   │   │   └── api.js     # API service layer
│   │   └── ...
│   └── package.json
│
├── server/                 # Express backend application
│   ├── src/
│   │   ├── server.js      # Express server setup
│   │   ├── routes/        # API routes
│   │   ├── controllers/   # Business logic
│   │   ├── middleware/    # Custom middleware
│   │   └── config/        # Database configuration
│   ├── uploads/           # Uploaded video files
│   └── package.json
│
└── README.md
```

## 🎯 Usage

### Uploading Videos

1. Click the **Upload** button in the header
2. Select one or more video files
3. Monitor upload progress
4. Videos will appear in your library automatically

### Searching Videos

- Use the search bar to find videos by:
  - Title
  - Description
  - Transcription content (for ML search)

### Managing Videos

- **View Details**: Click on any video card to open the detail modal
- **Edit**: Click the edit button in the video detail modal
- **Delete**: Click the delete button on any video card
- **Sort**: Use the sort dropdown to organize videos by:
  - Newest first
  - Oldest first
  - Most viewed
  - Title (A-Z)

### View Modes

Toggle between **Grid** and **List** view modes using the buttons in the toolbar.

## 🔌 API Endpoints

### Videos

- `GET /api/videos` - Get all videos (supports `search` and `sort` query params)
- `GET /api/videos/:id` - Get video by ID
- `POST /api/videos` - Upload a new video
- `PUT /api/videos/:id` - Update video metadata
- `DELETE /api/videos/:id` - Delete a video

### Health Check

- `GET /api/health` - Server health status

## 🎨 Customization

### Styling

The application uses Tailwind CSS. Customize the design by modifying:
- `client/src/index.css` - Global styles and animations
- `client/src/App.jsx` - Component styles (inline Tailwind classes)

### Color Scheme

The default color scheme uses red (`red-600`, `red-700`) as the primary color. You can customize this by updating the Tailwind classes throughout the application.

## 🚀 Production Deployment

### Build the Frontend

```bash
cd client
npm run build
```

The built files will be in the `client/dist` directory.

### Environment Setup

For production, make sure to:
1. Set proper environment variables
2. Configure CORS settings
3. Set up proper file storage (consider cloud storage like S3)
4. Implement AWS Transcribe for automatic transcription
5. Add authentication/authorization
6. Set up proper error logging

## 🔮 Future Enhancements

- [ ] AWS Transcribe integration for automatic transcription
- [ ] User authentication and authorization
- [ ] Video thumbnails generation
- [ ] Video streaming optimization
- [ ] Cloud storage integration (S3, etc.)
- [ ] Advanced search filters
- [ ] Video playlists
- [ ] Sharing and collaboration features
- [ ] Analytics dashboard

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on the repository.

---

Made with ❤️ using React and Express


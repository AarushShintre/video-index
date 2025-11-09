import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import videoRoutes from './routes/videos.js';
import semanticUpdateRoutes from './routes/semanticUpdate.js';
import { initDatabase } from './config/database.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));
app.use('/uploads', express.static(process.env.UPLOAD_DIR));



app.use('/api/videos', videoRoutes);

// ✅ both semantic routes properly mounted
app.use('/api/semantic-search', semanticUpdateRoutes);


app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

initDatabase().then(() => {
  app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
  });
});

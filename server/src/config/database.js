import sqlite3 from 'sqlite3';
import { promisify } from 'util';

const db = new sqlite3.Database(process.env.DATABASE_PATH || './database.sqlite');

db.run = promisify(db.run);
db.get = promisify(db.get);
db.all = promisify(db.all);

export const initDatabase = async () => {
  await db.run(`
    CREATE TABLE IF NOT EXISTS videos (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      description TEXT,
      filename TEXT NOT NULL,
      filepath TEXT NOT NULL,
      thumbnail TEXT,
      duration INTEGER DEFAULT 0,
      size INTEGER NOT NULL,
      type TEXT NOT NULL,
      views INTEGER DEFAULT 0,
      transcription TEXT,
      tags TEXT,
      created_at INTEGER NOT NULL
    )
  `);
  console.log('✅ Database initialized');
};

export default db;
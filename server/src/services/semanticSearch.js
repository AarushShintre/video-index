import { exec } from "child_process";
import path from "path";
import util from "util";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();
const execPromise = util.promisify(exec);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PYTHON = process.env.PYTHON_PATH || "python3";

const PYTHON_SCRIPT = path.join(
  process.cwd(),
  "src/services/semanticSearchHelper.py"
);

export async function addVideoToIndex(videoPath) {
  const resultsDir = path.join(__dirname, "results");

  const payload = JSON.stringify({
    operation: "add_to_index",
    video_path: videoPath,
    results_dir: resultsDir,
  });

  const cmd = `"${PYTHON}" "${PYTHON_SCRIPT}" '${payload}'`;
  console.log("🔍 Running:", cmd);

  try {
    const { stdout } = await execPromise(cmd);
    console.log("✅ Python Output:", stdout.trim());

    const jsonMatch = stdout.trim().match(/\{.*\}/);
    if (!jsonMatch) throw new Error("No JSON returned");

    return JSON.parse(jsonMatch[0]);
  } catch (err) {
    console.error("❌ Python Error:", err);
    throw err;
  }
}

import { exec } from "child_process";
import path from "path";
import dotenv from "dotenv";

dotenv.config();

const PYTHON_SCRIPT = path.join(
  process.cwd(),
  "src/services/semanticSearchHelper.py"
);

export const addVideoToIndex = (fullVideoPath) => {
  return new Promise((resolve, reject) => {
    const cmd = `python3 "${PYTHON_SCRIPT}" "${fullVideoPath}"`;

    console.log("🔍 Running:", cmd);

    exec(cmd, (error, stdout, stderr) => {
      if (error) {
        console.error("❌ Python Error:", stderr);
        return reject(error);
      }
      console.log("✅ Python Output:", stdout);
      resolve(stdout);
    });
  });
};

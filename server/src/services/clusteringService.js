import { spawn } from 'child_process';
import { promisify } from 'util';
import { exec } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join, resolve } from 'path';
import { existsSync } from 'fs';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Find Python executable (python3, python, or py313 for Windows)
 */
async function findPythonCommand() {
  // Try common Python commands, with Windows-specific ones first
  const commands = process.platform === 'win32' 
    ? ['python', 'py313', 'py -3.13', 'py -3', 'py', 'python3']
    : ['python3', 'python'];
  
  for (const cmd of commands) {
    try {
      // Use 'cmd /c' on Windows for commands with spaces (like 'py -3.13')
      const testCmd = process.platform === 'win32' && cmd.includes(' ')
        ? `cmd /c "${cmd} --version"`
        : `${cmd} --version`;
      
      await execAsync(testCmd);
      console.log(`Found Python: ${cmd}`);
      return cmd;
    } catch (error) {
      // Try next command
      continue;
    }
  }
  
  // Default fallback
  console.warn('Could not find Python, using "python" as fallback');
  return 'python';
}

/**
 * Run video clustering pipeline
 */
export async function runClusteringPipeline(uploadsDir, outputDir = 'output', options = {}) {
  const {
    maxVideos = 100,
    normalize = true,
    useCosine = true,
    skipFrames = 4,
    pcaComponents = 128,
    nClusters = 10
  } = options;

  try {
    const pythonCmd = await findPythonCommand();
    const projectRoot = resolve(__dirname, '../../..');
    const clusteringScript = join(projectRoot, 'video_clustering.py');
    
    if (!existsSync(clusteringScript)) {
      throw new Error(`Clustering script not found: ${clusteringScript}`);
    }

    // Ensure output directory exists
    const outputPath = resolve(projectRoot, outputDir);
    if (!existsSync(outputPath)) {
      const fs = await import('fs');
      fs.mkdirSync(outputPath, { recursive: true });
    }

    const args = [
      clusteringScript,
      '--data-dir', uploadsDir,
      '--output-dir', outputPath,
      '--max-videos', maxVideos.toString(),
      '--skip-frames', skipFrames.toString(),
      '--pca-components', pcaComponents.toString(),
      '--n-clusters', nClusters.toString()
    ];

    if (normalize) {
      args.push('--normalize');
    }

    if (useCosine) {
      args.push('--use-cosine');
    }

    console.log(`Starting clustering pipeline...`);
    console.log(`   Command: ${pythonCmd} ${args.join(' ')}`);

    return new Promise((resolve, reject) => {
      // Handle Windows commands with spaces (like 'py -3.13')
      let command = pythonCmd;
      let commandArgs = args;
      
      if (process.platform === 'win32' && pythonCmd.includes(' ')) {
        // Split command like 'py -3.13' into ['py', '-3.13']
        const parts = pythonCmd.split(' ');
        command = parts[0];
        commandArgs = [...parts.slice(1), ...args];
      }
      
      const python = spawn(command, commandArgs, {
        cwd: projectRoot,
        stdio: 'inherit', // This will show output in the console
        shell: process.platform === 'win32' // Use shell on Windows for better compatibility
      });

      let stdout = '';
      let stderr = '';

      python.stdout?.on('data', (data) => {
        const output = data.toString();
        stdout += output;
        process.stdout.write(output); // Show output in real-time
      });

      python.stderr?.on('data', (data) => {
        const output = data.toString();
        stderr += output;
        process.stderr.write(output); // Show errors in real-time
      });

      python.on('close', (code) => {
        if (code === 0) {
          console.log(`Clustering pipeline completed successfully`);
          resolve({ success: true, stdout, stderr });
        } else {
          console.error(`Clustering pipeline failed with code ${code}`);
          reject(new Error(`Clustering pipeline failed with exit code ${code}: ${stderr || stdout}`));
        }
      });

      python.on('error', (error) => {
        console.error(`Failed to spawn Python process:`, error);
        reject(new Error(`Failed to spawn Python process: ${error.message}`));
      });
    });
  } catch (error) {
    console.error('Error running clustering pipeline:', error);
    throw error;
  }
}

/**
 * Check if clustering results exist
 */
export function clusteringResultsExist(outputDir = 'output') {
  const projectRoot = resolve(__dirname, '../../../..');
  const outputPath = resolve(projectRoot, outputDir);
  
  const requiredFiles = [
    'clustering_results.npz',
    'faiss_index.bin',
    'video_embeddings.npz'
  ];

  return requiredFiles.every(file => {
    const filePath = join(outputPath, file);
    return existsSync(filePath);
  });
}


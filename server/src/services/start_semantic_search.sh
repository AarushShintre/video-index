#!/bin/bash
# Start the semantic search Python service

cd "$(dirname "$0")"
python semanticSearch.py --port 5001 --results-dir ../../output


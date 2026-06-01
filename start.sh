#!/bin/bash
# start.sh — launch the I2I recommender API + Gradio demo from this folder.
set -e
cd "$(dirname "$0")"

# Embeddings the demo/API serve. Category (cuisine) model by default;
# set I2I_EMBED_DIR=embeddings_i2i to fall back to the frozen baseline.
export I2I_EMBED_DIR="${I2I_EMBED_DIR:-embeddings_i2i_content}"
echo "Serving embeddings from: $I2I_EMBED_DIR"

echo "Starting API (uvicorn) on :8000 ..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# wait for the API to come up (model load can take ~30s)
echo "Waiting for API health ..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "API is up."; break
  fi
  sleep 2
done

echo "Starting Gradio demo on :7860 ..."
python demo.py &
DEMO_PID=$!

echo ""
echo "═══════════════════════════════════════════"
echo "  TableMind (I2I) is running"
echo "  API:  http://localhost:8000/docs"
echo "  Demo: http://localhost:7860"
echo "═══════════════════════════════════════════"
echo "  ngrok http 7860   # for external access"
echo ""

trap "kill $API_PID $DEMO_PID 2>/dev/null" EXIT
wait $DEMO_PID

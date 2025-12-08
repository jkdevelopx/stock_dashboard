#!/usr/bin/env bash
# start.sh
# ensure executable: chmod +x start.sh

# load env from .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# run streamlit
streamlit run mcrf_auto_scanner.py --server.port $PORT --server.address 0.0.0.0
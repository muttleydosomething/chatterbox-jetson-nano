#!/bin/bash
# Apply cuFFT CPU fallback before starting server
export PYTHONPATH=/app/patches:$PYTHONPATH
python3 /app/server.py

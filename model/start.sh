#!/bin/bash
# Start script for Render deployment

# Get port from environment variable (Render sets this)
PORT=${PORT:-5001}

# Start Gunicorn with Flask app
exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 300 app:app


#!/bin/bash
# Start TEFAS Analysis Full Stack
# Usage: ./run.sh

echo "========================================"
echo "TEFAS Fund Analysis - Full Stack"
echo "========================================"

# Kill any existing processes
echo "Stopping existing processes..."
pkill -f "uvicorn api.server" 2>/dev/null || true

# Start backend
echo ""
echo "[1/2] Starting Backend API (port 8000)..."
cd "$(dirname "$0")"
source venv/bin/activate
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✓ Backend is running at http://localhost:8000"
else
    echo "✗ Backend failed to start"
    exit 1
fi

# Start frontend
echo ""
echo "[2/2] Starting Frontend (port 5173)..."
cd ../tefas-insight
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "========================================"
echo "Services Started:"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Frontend: http://localhost:5173"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait and cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

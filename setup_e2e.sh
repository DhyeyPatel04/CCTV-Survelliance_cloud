#!/bin/bash
# Run this once on the E2E instance after uploading the project
set -e

echo "=== Installing system deps ==="
sudo apt-get update -q
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0

echo "=== Setting up Python venv ==="
cd backend
python3 -m venv venv_backend
source venv_backend/bin/activate

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing requirements ==="
pip install -r requirements.txt

echo "=== Building frontend ==="
cd ../frontend
npm install
npm run build

echo "=== Creating .env (fill in your Telegram credentials) ==="
cd ../backend
if [ ! -f .env ]; then
  cat > .env << 'EOF'
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
EOF
  echo "Created backend/.env — edit it with your Telegram credentials"
fi

echo ""
echo "=== Setup complete ==="
echo "Start the server with:"
echo "  cd backend && source venv_backend/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000"

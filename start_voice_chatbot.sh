#!/bin/bash
# LangChain Voice Chatbot Startup Script

echo "🚀 Starting LangChain Voice Chatbot..."
echo "Make sure Ollama is running with: ollama serve"
echo "Required model: gpt-oss:20b"
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama server is not running!"
    echo "Please start it with: ollama serve"
    exit 1
fi

# Check if model exists
if ! ollama list | grep -q "gpt-oss:20b"; then
    echo "❌ Model gpt-oss:20b not found!"
    echo "Please install it with: ollama pull gpt-oss:20b"
    exit 1
fi

echo "✅ Ollama server is running"
echo "✅ Model gpt-oss:20b is available"
echo ""

# Activate virtual environment and start chatbot
cd "$(dirname "$0")"
source .venv/bin/activate
echo "🎤 Starting voice chatbot..."
python langchain_voice_chatbot.py

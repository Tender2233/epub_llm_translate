#!/bin/bash
# Quick start script for EPUB Translator

echo "==================================="
echo "EPUB Translator - Quick Start Setup"
echo "==================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Check for config file
if [ ! -f "config.json" ]; then
    echo "⚠️  config.json not found"
    echo ""
    echo "Creating config.json from template..."
    cp config.template.json config.json
    echo "✓ Created config.json"
    echo ""
    echo "📝 Please edit config.json and add your API key:"
    echo "   - For Kimi: Get API key from https://platform.moonshot.cn"
    echo "   - For Claude: Get API key from https://console.anthropic.com"
    echo ""
    echo "Example config for Kimi (recommended):"
    echo '  {'
    echo '    "api_provider": "kimi",'
    echo '    "kimi": {'
    echo '      "api_key": "your-kimi-api-key-here",'
    echo '      "model": "moonshot-v1-128k"'
    echo '    }'
    echo '  }'
    echo ""
else
    echo "✓ config.json found"
    
    # Check if API key is configured
    if grep -q '"api_key": ""' config.json; then
        echo "⚠️  API key not configured in config.json"
        echo "   Please edit config.json and add your API key"
    else
        echo "✓ API key configured"
    fi
fi

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Usage:"
echo "  python3 translate_epub.py input.epub output_chinese.epub"
echo ""
echo "For more options:"
echo "  python3 translate_epub.py --help"
echo ""

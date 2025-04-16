#!/bin/bash

# CV Extractor with Open-Source LLMs using Ollama
# Project Runner Script

echo "=========================================="
echo "CV Extractor with Open-Source LLMs Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is required but not installed."
    exit 1
fi

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null; then
    echo "Installing virtualenv..."
    pip install virtualenv
fi

# Check if Ollama is installed and running
echo "Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "Warning: Ollama is not installed or not in PATH."
    echo "Please install Ollama from https://ollama.ai/"
    echo "After installation, run: ollama serve"
else
    echo "Ollama is installed."
    
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "Warning: Ollama server is not running."
        echo "Please start Ollama with: ollama serve"
    else
        echo "Ollama server is running."
        
        # Check for required models
        echo "Checking for required models..."
        MODELS=("llama3.2" "mistral" "qwen2.5")
        MISSING_MODELS=()
        
        for model in "${MODELS[@]}"; do
            if ! curl -s http://localhost:11434/api/tags | grep -q "$model"; then
                MISSING_MODELS+=("$model")
            fi
        done
        
        if [ ${#MISSING_MODELS[@]} -eq 0 ]; then
            echo "All required models are installed."
        else
            echo "The following models need to be pulled:"
            for missing in "${MISSING_MODELS[@]}"; do
                echo "  - $missing"
            done
            
            echo "Would you like to pull the missing models now? (y/n)"
            read -r PULL_MODELS
            
            if [[ $PULL_MODELS == "y" || $PULL_MODELS == "Y" ]]; then
                for missing in "${MISSING_MODELS[@]}"; do
                    echo "Pulling $missing model..."
                    ollama pull "$missing"
                done
            else
                echo "Please pull the missing models manually with: ollama pull <model-name>"
            fi
        fi
    fi
fi

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    virtualenv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads results comparisons data/ground_truth data/results

# Run the Flask application
echo "Starting the Flask application..."
python app.py

# Deactivate virtual environment when done
deactivate
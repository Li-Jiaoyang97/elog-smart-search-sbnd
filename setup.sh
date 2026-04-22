#!/usr/bin/env bash

# Exit on error
set -e

ENV_NAME="venv-semanticsearch"

# Avoid recreating if it already exists
if [ -d "$ENV_NAME" ]; then
    echo "Virtual environment '$ENV_NAME' already exists, skipping creation..."
else
    echo "Creating virtual environment..."
    python3.11 -m venv "$ENV_NAME"
fi

echo "Activating virtual environment..."
source "$ENV_NAME/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

if [ -f requirements.txt ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found, skipping..."
fi

MODEL_DIR="ElogSemanticSearch/models/all-mpnet-base-v2"
if [ -d "$MODEL_DIR" ]; then
    echo "Model already exists, skipping download..."
else
    echo "Downloading sentence-transformers model..."
    python3.11 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2').save('$MODEL_DIR')"
fi

echo ""
echo "Setup complete! To activate later, run:"
echo "  source $ENV_NAME/bin/activate"

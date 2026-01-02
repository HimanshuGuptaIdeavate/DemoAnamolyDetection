#!/bin/bash

###############################################################################
# WATCHTOWER - Install Validation & MLflow Components
# Places validation files in correct project locations
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "📦 Installing WATCHTOWER Validation & MLflow Components"
echo "========================================================"
echo ""

# Check we're in project root
if [ ! -d "src/watchtower" ]; then
    echo "❌ Error: Must run from project root (DemoAnamolyDetection/)"
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Create directories if they don't exist
echo -e "${BLUE}[1/4]${NC} Creating directories..."
mkdir -p src/watchtower/data
mkdir -p src/watchtower/utils
mkdir -p scripts
mkdir -p reports/{validation,plots}
echo "  ✓ Directories created"

# Copy validation files
echo -e "${BLUE}[2/4]${NC} Installing validation files..."
cp validation_config.py src/watchtower/data/
cp validate_data.py src/watchtower/data/
echo "  ✓ Validation suite installed"

# Copy MLflow logger
echo -e "${BLUE}[3/4]${NC} Installing MLflow logger..."
cp mlflow_logger.py src/watchtower/utils/
echo "  ✓ MLflow integration installed"

# Copy master script
echo -e "${BLUE}[4/4]${NC} Installing pipeline script..."
cp run_validation_pipeline.py scripts/
chmod +x scripts/run_validation_pipeline.py
echo "  ✓ Pipeline script installed"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📁 Files installed:"
echo "   src/watchtower/data/validation_config.py"
echo "   src/watchtower/data/validate_data.py"
echo "   src/watchtower/utils/mlflow_logger.py"
echo "   scripts/run_validation_pipeline.py"
echo ""
echo "🚀 Next steps:"
echo "   1. Activate environment: source .venv/bin/activate"
echo "   2. Run validation: python scripts/run_validation_pipeline.py"
echo "   3. View MLflow UI: mlflow ui"
echo ""

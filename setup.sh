#!/usr/bin/env bash
# setup.sh — One-shot environment setup for EDGAR Multi-Agent Comparator
# Works on WSL2 / Ubuntu / macOS.
#
# Usage:  chmod +x setup.sh && ./setup.sh

set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   EDGAR Multi-Agent Comparator — Environment Setup  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Check prerequisites ────────────────────────────────────────────────────

command -v az   >/dev/null 2>&1 || { echo "❌ Azure CLI not found. Install: https://aka.ms/installazureclilinux"; exit 1; }
command -v uv   >/dev/null 2>&1 || { echo "❌ uv not found. Install: curl -Lsf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 not found"; exit 1; }

echo "✓  Azure CLI:  $(az --version | head -1)"
echo "✓  uv:         $(uv --version)"
echo "✓  Python:     $(python3 --version)"
echo ""

# ── 2. Azure login check ──────────────────────────────────────────────────────

echo "── Azure Authentication ─────────────────────────────────"
if az account show >/dev/null 2>&1; then
    SUBSCRIPTION=$(az account show --query name -o tsv)
    echo "✓  Logged in to: $SUBSCRIPTION"
else
    echo "→  Logging in to Azure..."
    az login --use-device-code
fi
echo ""

# ── 3. Install Python dependencies ───────────────────────────────────────────

echo "── Installing Python dependencies ──────────────────────"
uv sync 2>/dev/null || uv pip install -r requirements.txt
echo "✓  Dependencies installed"
echo ""

# ── 4. Validate .env ─────────────────────────────────────────────────────────

echo "── Validating .env ──────────────────────────────────────"
REQUIRED_VARS=("RESOURCE_GROUP" "FOUNDRY_PROJECT_ENDPOINT" "SEARCH_ENDPOINT" "SEARCH_KEY")
MISSING=0

for var in "${REQUIRED_VARS[@]}"; do
    value=$(grep -E "^${var}=" .env 2>/dev/null | cut -d= -f2- | tr -d ' \t\r' | sed 's/#.*//')
    if [ -z "$value" ]; then
        echo "  ❌ Missing: $var"
        MISSING=$((MISSING + 1))
    else
        echo "  ✓  $var is set"
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "⚠  Fix $MISSING missing variable(s) in .env then re-run setup.sh"
    exit 1
fi
echo ""

# ── 5. (Optional) Deploy supplemental resources via Bicep ────────────────────

RG=$(grep -E "^RESOURCE_GROUP=" .env | cut -d= -f2- | tr -d ' \t\r' | sed 's/#.*//')

echo "── Bicep Deployment (Cosmos DB + Storage) ───────────────"
read -p "Deploy Cosmos DB and Storage to '$RG'? [y/N] " -n 1 -r DEPLOY
echo ""

if [[ $DEPLOY =~ ^[Yy]$ ]]; then
    echo "→  Deploying main.bicep to $RG..."
    DEPLOYMENT_OUTPUT=$(az deployment group create \
        --resource-group "$RG" \
        --template-file main.bicep \
        --parameters deploySearch=false \
        --output json)

    COSMOS_ENDPOINT=$(echo "$DEPLOYMENT_OUTPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['properties']['outputs'].get('cosmosEndpoint',{}).get('value',''))" 2>/dev/null || echo "")
    STORAGE_CONN=$(echo "$DEPLOYMENT_OUTPUT"   | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['properties']['outputs'].get('storageConnectionString',{}).get('value',''))" 2>/dev/null || echo "")

    if [ -n "$COSMOS_ENDPOINT" ]; then
        echo "  → Updating .env with Cosmos endpoint..."
        # Append if not already present
        grep -q "^COSMOS_ENDPOINT=" .env && \
            sed -i "s|^COSMOS_ENDPOINT=.*|COSMOS_ENDPOINT=$COSMOS_ENDPOINT|" .env || \
            echo "COSMOS_ENDPOINT=$COSMOS_ENDPOINT" >> .env
    fi

    if [ -n "$STORAGE_CONN" ]; then
        echo "  → Updating .env with Storage connection string..."
        grep -q "^STORAGE_CONNECTION_STRING=" .env && \
            sed -i "s|^STORAGE_CONNECTION_STRING=.*|STORAGE_CONNECTION_STRING=$STORAGE_CONN|" .env || \
            echo "STORAGE_CONNECTION_STRING=$STORAGE_CONN" >> .env
    fi

    echo "✓  Bicep deployment complete"
else
    echo "  Skipping Bicep deployment — you can run it manually later."
fi
echo ""

# ── 6. Create data directories ────────────────────────────────────────────────

mkdir -p data evaluation
echo "✓  Directories: data/, evaluation/"
echo ""

# ── 7. Summary ────────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete!  Next steps:                       ║"
echo "║                                                      ║"
echo "║  1. Download & index EDGAR filings:                  ║"
echo "║     uv run python -m ingestion.fetch_and_index       ║"
echo "║                                                      ║"
echo "║  2. Create agents:                                   ║"
echo "║     uv run python -m agents.setup_agents --setup     ║"
echo "║                                                      ║"
echo "║  3. Launch the Streamlit app:                        ║"
echo "║     uv run streamlit run app.py                      ║"
echo "║                                                      ║"
echo "║  4. Run evaluations:                                 ║"
echo "║     uv run python -m evaluation.run_evals            ║"
echo "╚══════════════════════════════════════════════════════╝"

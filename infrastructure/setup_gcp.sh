#!/bin/bash
# GCP Infrastructure Setup for Circadian Analytics Dashboard
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PROJECT_ID="sleep-health-dashboard"
REGION="us-central1"
BUCKET_NAME="sleep-health-dashboard-healthkit-exports"
FUNCTION_NAME="healthkit-ingest"

echo "=============================================="
echo "Circadian Analytics Dashboard - GCP Setup"
echo "=============================================="
echo ""

# Set project
echo "1. Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo ""
echo "2. Enabling required APIs..."
gcloud services enable \
    cloudfunctions.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    iap.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com

# Create GCS bucket
echo ""
echo "3. Creating GCS bucket..."
if gsutil ls -b gs://${BUCKET_NAME} &>/dev/null; then
    echo "   Bucket already exists: gs://${BUCKET_NAME}"
else
    gsutil mb -l ${REGION} -c STANDARD gs://${BUCKET_NAME}
    gsutil uniformbucketlevelaccess set on gs://${BUCKET_NAME}
    echo "   Created bucket: gs://${BUCKET_NAME}"
fi

# Generate API key
echo ""
echo "4. Generating API key..."
API_KEY="hk_$(openssl rand -hex 16)"
mkdir -p "${PROJECT_ROOT}/credentials"
echo "${API_KEY}" > "${PROJECT_ROOT}/credentials/api_key.txt"
echo "   API key saved to credentials/api_key.txt"

# Deploy Cloud Function
echo ""
echo "5. Deploying Cloud Function..."

gcloud functions deploy ${FUNCTION_NAME} \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source="${PROJECT_ROOT}/cloud_function" \
    --entry-point=healthkit_ingest \
    --trigger-http \
    --allow-unauthenticated \
    --memory=256MB \
    --timeout=60s \
    --set-env-vars="API_KEY=${API_KEY},GCS_BUCKET=${BUCKET_NAME}"

# Get function URL
FUNCTION_URL=$(gcloud functions describe ${FUNCTION_NAME} --region=${REGION} --gen2 --format='value(serviceConfig.uri)')

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Cloud Function URL:"
echo "  ${FUNCTION_URL}"
echo ""
echo "API Key:"
echo "  ${API_KEY}"
echo ""
echo "=============================================="
echo "Auto Export App Configuration"
echo "=============================================="
echo ""
echo "1. Open Auto Export on iPhone"
echo "2. Go to: Destinations → Add REST API"
echo "3. Configure:"
echo "   - URL: ${FUNCTION_URL}"
echo "   - Method: POST"
echo "   - Add Header:"
echo "       Name: X-API-Key"
echo "       Value: ${API_KEY}"
echo "4. Test the connection"
echo ""
echo "=============================================="
echo "Next Steps"
echo "=============================================="
echo "1. Configure Auto Export app with the settings above"
echo "2. Run infrastructure/deploy_cloudrun.sh to deploy the dashboard"
echo "3. Enable IAP in GCP Console for the Cloud Run service"
echo ""

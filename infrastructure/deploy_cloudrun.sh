#!/bin/bash
# Deploy Circadian Dashboard to Cloud Run
set -e

PROJECT_ID="sleep-health-dashboard"
REGION="us-central1"
SERVICE_NAME="circadian-dashboard"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
BUCKET_NAME="sleep-health-dashboard-healthkit-exports"

echo "=============================================="
echo "Deploying Circadian Dashboard to Cloud Run"
echo "=============================================="
echo ""

cd "$(dirname "$0")/.."

echo "1. Building container image..."
gcloud builds submit --tag ${IMAGE_NAME}

echo ""
echo "2. Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --region ${REGION} \
    --platform managed \
    --memory 1Gi \
    --min-instances 0 \
    --max-instances 2 \
    --no-allow-unauthenticated \
    --set-env-vars "GCS_BUCKET=${BUCKET_NAME}"

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)')

echo ""
echo "=============================================="
echo "Deployment Complete!"
echo "=============================================="
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "=============================================="
echo "NEXT STEPS (Manual in GCP Console)"
echo "=============================================="
echo ""
echo "1. Enable IAP:"
echo "   https://console.cloud.google.com/security/iap?project=${PROJECT_ID}"
echo ""
echo "2. Find '${SERVICE_NAME}' in the list"
echo ""
echo "3. Toggle IAP ON"
echo ""
echo "4. Click 'Add Principal' and add:"
echo "   - Your Google account email"
echo "   - Role: 'IAP-secured Web App User'"
echo ""
echo "5. Visit ${SERVICE_URL} and authenticate with Google"
echo ""

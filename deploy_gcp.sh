#!/bin/bash
export AWS_ACCESS_KEY_ID=<AWS-AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
gcloud run deploy sentiment-app \
  --image=asia-southeast1-docker.pkg.dev/ai-projects-476902/class/sentiment:latest \
  --platform=managed \
  --region=asia-southeast1 \
  --port=8000 \
  --cpu=2 \
  --memory=4Gi \
  --allow-unauthenticated \
  --set-env-vars=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID",AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
#!/bin/bash
# Grant the GKE node service account access to the compilation cache bucket.
#
# The pods run as the "default" KSA, which is bound to the GKE node GSA.
# We need storage.objects.{list,get,create} on gs://sivaibhav-exp
#
# Run this once from a machine with gcloud + project owner/editor access.

set -euo pipefail

GCP_PROJECT="cloud-tpu-multipod-dev"
BUCKET="sivaibhav-exp"

# Look up the numeric project number — the default Compute Engine SA uses this,
# not the project ID string.
PROJECT_NUMBER=$(gcloud projects describe "${GCP_PROJECT}" --format='value(projectNumber)')
echo "Project: ${GCP_PROJECT}  Number: ${PROJECT_NUMBER}"

# Default Compute Engine service account: <number>-compute@developer.gserviceaccount.com
NODE_GSA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Granting storage.objectAdmin on gs://${BUCKET} to ${NODE_GSA} ..."

gsutil iam ch \
  serviceAccount:${NODE_GSA}:roles/storage.objectAdmin \
  gs://${BUCKET}

echo "Done. New IAM policy:"
gsutil iam get gs://${BUCKET} | grep -A2 "${NODE_GSA}"

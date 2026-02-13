# PowerShell Setup Script for Google Cloud Batch
$PROJECT_ID = "seminar-qf"
$REGION = "us-central1"
$BUCKET_NAME = "seminar-qf-batch-data-001"
$REPO_NAME = "batch-images"
$IMAGE_NAME = "monte-carlo-garch"
$IMAGE_TAG = "latest"
$IMAGE_URI = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME`:$IMAGE_TAG"
$MODEL = "garch"  # garch | regime-switching | ms-garch

# Copy/paste run commands (submit directly without editing this script):
# gcloud batch jobs submit "garch-10k-$(Get-Date -Format 'yyyyMMdd-HHmm')" --location us-central1 --config batch/job_garch_10k.json

# gcloud batch jobs submit "rs-10k-$(Get-Date -Format 'yyyyMMdd-HHmm')" --location us-central1 --config batch/job_regime_switching_10k.json

# gcloud batch jobs submit "msgarch-10k-$(Get-Date -Format 'yyyyMMdd-HHmm')" --location us-central1 --config batch/job_ms_garch_10k.json

Write-Host "--- Google Cloud Batch Setup ---" -ForegroundColor Cyan
Write-Host "Project: $PROJECT_ID"
Write-Host "Region: $REGION"
Write-Host "Bucket: $BUCKET_NAME"
Write-Host "Image: $IMAGE_URI"
Write-Host "--------------------------------"

# 1. Enable Required Services (API calls can take a moment)
Write-Host "`n[1/6] Enabling services..."
gcloud services enable batch.googleapis.com compute.googleapis.com logging.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# 2. Create Storage Bucket
Write-Host "`n[2/6] Creating/Checking Storage Bucket..."
if (gsutil ls -b gs://$BUCKET_NAME 2>$null) {
    Write-Host "Bucket gs://$BUCKET_NAME already exists." -ForegroundColor Yellow
} else {
    gcloud storage buckets create gs://$BUCKET_NAME --location=$REGION
    Write-Host "Bucket created." -ForegroundColor Green
}

# 3. Create Artifact Registry
Write-Host "`n[3/6] Creating/Checking Artifact Registry..."
$repoExists = gcloud artifacts repositories list --location=$REGION --filter="name:$REPO_NAME" --format="value(name)"
if ($repoExists) {
    Write-Host "Repository $REPO_NAME already exists." -ForegroundColor Yellow
} else {
    gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=$REGION --description="Docker repository for Batch jobs"
    Write-Host "Repository created." -ForegroundColor Green
}

# 4. Upload Input Data
Write-Host "`n[4/6] Uploading Input Data..."
# Ensure we are in the root correct directory relative to the script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$InputFile = "data/output/daily_asset_returns_with_garch.csv"
$InputFileRS = "data/output/daily_asset_returns_with_regime.csv"
$InputFileMSG = "data/output/daily_asset_returns_with_msgarch.csv"
$MertonFile = "data/output/merged_data_with_merton.csv"

if (Test-Path $InputFile) {
    gcloud storage cp $InputFile gs://$BUCKET_NAME/data/output/
    Write-Host "Uploaded $InputFile" -ForegroundColor Green
} else {
    Write-Error "Input file not found: $InputFile"
}

if (Test-Path $InputFileRS) {
    gcloud storage cp $InputFileRS gs://$BUCKET_NAME/data/output/
    Write-Host "Uploaded $InputFileRS" -ForegroundColor Green
} else {
    Write-Warning "Input file not found: $InputFileRS"
}

if (Test-Path $InputFileMSG) {
    gcloud storage cp $InputFileMSG gs://$BUCKET_NAME/data/output/
    Write-Host "Uploaded $InputFileMSG" -ForegroundColor Green
} else {
    Write-Warning "Input file not found: $InputFileMSG"
}

if (Test-Path $MertonFile) {
    gcloud storage cp $MertonFile gs://$BUCKET_NAME/data/output/
    Write-Host "Uploaded $MertonFile" -ForegroundColor Green
} else {
    Write-Warning "Merton file not found: $MertonFile. Proceeding without it (PD calculation might be limited)."
}

# 5. Build and Push Docker Image (Using Cloud Build)
Write-Host "`n[5/6] Building and Pushing Docker Image (via Cloud Build)..."

# Submit build to Cloud Build using the YAML config
# This allows us to specify the Dockerfile location in a subdirectory (batch/Dockerfile)
gcloud builds submit --config batch/cloudbuild.yaml --substitutions _IMAGE_URI=$IMAGE_URI .

if ($LASTEXITCODE -eq 0) {
    Write-Host "Image built and pushed successfully." -ForegroundColor Green
} else {
    Write-Error "Cloud Build failed."
    exit 1
}

# 6. Submit Batch Job
Write-Host "`n[6/6] Submitting Batch Job..."
$jobConfig = "batch/job_garch_10k.json"
if ($MODEL -eq "regime-switching") {
    $jobConfig = "batch/job_regime_switching_10k.json"
} elseif ($MODEL -eq "ms-garch") {
    $jobConfig = "batch/job_ms_garch_10k.json"
}

gcloud batch jobs submit "monte-carlo-run-$(Get-Date -Format 'yyyyMMdd-HHmm')" `
    --location $REGION `
    --config $jobConfig

Write-Host "`n--- Setup Complete! Check the Google Cloud Console for job status. ---" -ForegroundColor Cyan

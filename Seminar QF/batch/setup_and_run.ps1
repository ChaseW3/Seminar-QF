# PowerShell Setup Script for Google Cloud Batch
$PROJECT_ID = "seminar-qf"
$REGION = "us-central1"
$BUCKET_NAME = "seminar-qf-batch-data-001"
$REPO_NAME = "batch-images"
$IMAGE_NAME = "monte-carlo-garch"
$IMAGE_TAG = "latest"
$IMAGE_URI = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME`:$IMAGE_TAG"
$MODEL = "all-1k"  # all-1k | merton | regime-switching | ms-garch
$RUN_ID = (Get-Date -Format "yyyyMMdd-HHmmss")
$INPUT_PREFIX = "data/releases/$RUN_ID"
$OUTPUT_PREFIX = "output/runs/$RUN_ID/results_1k"

# This script uploads a fresh release snapshot under gs://$BUCKET_NAME/$INPUT_PREFIX/
# and submits model jobs with output under gs://$BUCKET_NAME/$OUTPUT_PREFIX/<model>/

Write-Host "--- Google Cloud Batch Setup ---" -ForegroundColor Cyan
Write-Host "Project: $PROJECT_ID"
Write-Host "Region: $REGION"
Write-Host "Bucket: $BUCKET_NAME"
Write-Host "Image: $IMAGE_URI"
Write-Host "Run ID: $RUN_ID"
Write-Host "Input Prefix: $INPUT_PREFIX"
Write-Host "Output Prefix: $OUTPUT_PREFIX"
Write-Host "--------------------------------"

# 1. Enable Required Services (API calls can take a moment)
Write-Host "`n[1/6] Enabling services..."
gcloud services enable batch.googleapis.com compute.googleapis.com logging.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# 2. Create Storage Bucket
Write-Host "`n[2/6] Creating/Checking Storage Bucket..."
$null = gsutil ls -b "gs://$BUCKET_NAME" 2>$null
if ($LASTEXITCODE -eq 0) {
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

$InputFileRS = "data/output/daily_asset_returns_with_regime_switching.csv"
$InputFileMSG = "data/output/daily_asset_returns_with_msgarch.csv"
$MertonFile = "data/output/merged_data_with_merton.csv"

if (Test-Path $InputFileRS) {
    gcloud storage cp $InputFileRS gs://$BUCKET_NAME/$INPUT_PREFIX/
    Write-Host "Uploaded $InputFileRS" -ForegroundColor Green
} else {
    Write-Error "Required input file not found: $InputFileRS"
    exit 1
}

if (Test-Path $InputFileMSG) {
    gcloud storage cp $InputFileMSG gs://$BUCKET_NAME/$INPUT_PREFIX/
    Write-Host "Uploaded $InputFileMSG" -ForegroundColor Green
} else {
    Write-Error "Required input file not found: $InputFileMSG"
    exit 1
}

if (Test-Path $MertonFile) {
    gcloud storage cp $MertonFile gs://$BUCKET_NAME/$INPUT_PREFIX/
    Write-Host "Uploaded $MertonFile" -ForegroundColor Green
} else {
    Write-Error "Required Merton file not found: $MertonFile"
    exit 1
}

# Optional release manifest for traceability (prevents accidental old-input ambiguity)
$manifestPath = "batch_input_manifest_$RUN_ID.txt"
@(
    "run_id=$RUN_ID"
    "region=$REGION"
    "bucket=$BUCKET_NAME"
    "input_prefix=$INPUT_PREFIX"
    "output_prefix=$OUTPUT_PREFIX"
    "regime_file=$InputFileRS"
    "ms_garch_file=$InputFileMSG"
    "merton_file=$MertonFile"
) | Set-Content -Path $manifestPath
gcloud storage cp $manifestPath gs://$BUCKET_NAME/$INPUT_PREFIX/
Write-Host "Uploaded input manifest: $manifestPath" -ForegroundColor Green

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

# 6. Submit Batch Jobs (1k) with run-specific input/output paths
Write-Host "`n[6/6] Submitting Batch Job(s)..."

New-Item -ItemType Directory -Path "batch/generated" -Force | Out-Null

function New-RunScopedJobConfig {
    param(
        [string]$TemplatePath,
        [string]$OutputPath,
        [string]$InputFileName,
        [string]$ModelOutputSuffix
    )

    $jobObj = Get-Content -Raw -Path $TemplatePath | ConvertFrom-Json

    $commands = @($jobObj.taskGroups[0].taskSpec.runnables[0].container.commands)

    function Set-CommandArgValue {
        param(
            [object[]]$Cmd,
            [string]$Flag,
            [string]$Value
        )
        $idx = [Array]::IndexOf($Cmd, $Flag)
        if ($idx -lt 0 -or ($idx + 1) -ge $Cmd.Length) {
            throw "Could not find flag '$Flag' in job template commands"
        }
        $Cmd[$idx + 1] = $Value
        return $Cmd
    }

    $commands = Set-CommandArgValue -Cmd $commands -Flag "--input-file" -Value "$INPUT_PREFIX/$InputFileName"
    $commands = Set-CommandArgValue -Cmd $commands -Flag "--merton-file" -Value "$INPUT_PREFIX/merged_data_with_merton.csv"
    $commands = Set-CommandArgValue -Cmd $commands -Flag "--output-prefix" -Value "$OUTPUT_PREFIX/$ModelOutputSuffix"

    $jobObj.taskGroups[0].taskSpec.runnables[0].container.commands = $commands
    $jobObj | ConvertTo-Json -Depth 30 | Set-Content -Path $OutputPath
}

function Submit-ModelJob {
    param(
        [string]$Model,
        [string]$Template,
        [string]$InputFileName,
        [string]$Suffix,
        [string]$JobNamePrefix
    )

    $generatedConfig = "batch/generated/job_${Model}_1k_$RUN_ID.json"
    New-RunScopedJobConfig -TemplatePath $Template -OutputPath $generatedConfig -InputFileName $InputFileName -ModelOutputSuffix $Suffix

    $jobName = "$JobNamePrefix-$RUN_ID"
    gcloud batch jobs submit $jobName --location $REGION --config $generatedConfig
    Write-Host "Submitted $Model with config: $generatedConfig" -ForegroundColor Green
}

if ($MODEL -eq "all-1k") {
    Submit-ModelJob -Model "merton" -Template "batch/job_merton_1k.json" -InputFileName "merged_data_with_merton.csv" -Suffix "merton" -JobNamePrefix "merton-1k"
    Submit-ModelJob -Model "regime-switching" -Template "batch/job_regime_switching_1k.json" -InputFileName "daily_asset_returns_with_regime_switching.csv" -Suffix "regime-switching" -JobNamePrefix "rs-1k"
    Submit-ModelJob -Model "ms-garch" -Template "batch/job_ms_garch_1k.json" -InputFileName "daily_asset_returns_with_msgarch.csv" -Suffix "ms-garch" -JobNamePrefix "msgarch-1k"
} elseif ($MODEL -eq "merton") {
    Submit-ModelJob -Model "merton" -Template "batch/job_merton_1k.json" -InputFileName "merged_data_with_merton.csv" -Suffix "merton" -JobNamePrefix "merton-1k"
} elseif ($MODEL -eq "regime-switching") {
    Submit-ModelJob -Model "regime-switching" -Template "batch/job_regime_switching_1k.json" -InputFileName "daily_asset_returns_with_regime_switching.csv" -Suffix "regime-switching" -JobNamePrefix "rs-1k"
} elseif ($MODEL -eq "ms-garch") {
    Submit-ModelJob -Model "ms-garch" -Template "batch/job_ms_garch_1k.json" -InputFileName "daily_asset_returns_with_msgarch.csv" -Suffix "ms-garch" -JobNamePrefix "msgarch-1k"
} else {
    Write-Error "Unsupported MODEL: $MODEL"
    exit 1
}

Write-Host "`n--- Setup Complete! Check the Google Cloud Console for job status. ---" -ForegroundColor Cyan

param(
  [Parameter(Mandatory = $true)] [string]$ProjectId,
  [Parameter(Mandatory = $true)] [string]$Region,
  [Parameter(Mandatory = $true)] [string]$BucketName,
  [string]$RepoName = "aih-scraper-repo",
  [string]$JobName = "senior-activities-scraper",
  [string]$VertexModel = "gemini-3.1-pro-preview",
  [string]$VertexLocation = "global",
  [bool]$EnableLlmTagging = $true,
  [int]$LlmParseFailureDisableThreshold = 40
)

$ErrorActionPreference = "Stop"

$Image = "$Region-docker.pkg.dev/$ProjectId/$RepoName/${JobName}:latest"
$EnableLlmTaggingValue = if ($EnableLlmTagging) { "true" } else { "false" }
$EnvVars = "GCS_BUCKET_NAME=$BucketName,ENABLE_LLM_TAGGING=$EnableLlmTaggingValue,LLM_PARSE_FAILURE_DISABLE_THRESHOLD=$LlmParseFailureDisableThreshold,VERTEX_PROJECT=$ProjectId,VERTEX_LOCATION=$VertexLocation,VERTEX_MODEL=$VertexModel"

Write-Host "Using image: $Image"

& gcloud.cmd config set project $ProjectId
& gcloud.cmd services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# Create Artifact Registry repo if it doesn't exist
& gcloud.cmd artifacts repositories describe $RepoName --location=$Region 2>$null
if ($LASTEXITCODE -ne 0) {
  & gcloud.cmd artifacts repositories create $RepoName --repository-format=docker --location=$Region --description="Repo for scraper job"
}

Push-Location $PSScriptRoot
try {
  & gcloud.cmd builds submit --tag $Image .
} finally {
  Pop-Location
}

& gcloud.cmd run jobs describe $JobName --region=$Region 2>$null
if ($LASTEXITCODE -eq 0) {
  & gcloud.cmd run jobs update $JobName --image $Image --region=$Region --set-env-vars $EnvVars
} else {
  & gcloud.cmd run jobs create $JobName --image $Image --region=$Region --max-retries=1 --task-timeout=1800s --set-env-vars $EnvVars
}

& gcloud.cmd run jobs execute $JobName --region=$Region

Write-Host "Done. Check outputs in gs://$BucketName/"

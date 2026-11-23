# Deploying BookForge

BookForge is a **long-running batch pipeline** (minutes to hours per channel).
That drives the deployment choice: prefer a *job* runtime over a request-driven
HTTP service.

## Topic-to-Book Web App

The interactive topic-to-book web app (`bookforge/web/app.py`) deploys to Vercel
and is documented in the root [README](../README.md). For the batch YouTube
pipeline below, read on.

## Prereqs

- GCP project with billing, `gcloud` authenticated
- APIs: `run`, `artifactregistry`, `secretmanager`
- An EURI API key: get a Bearer key from <https://euron.one/euri>

## Option A — Cloud Run Job (recommended)

```bash
export PROJECT_ID=<your-project> REGION=us-central1
gcloud config set project $PROJECT_ID

# 1) Secret for the API key (never in the image or logs)
printf '%s' "$EURI_API_KEY" | gcloud secrets create bookforge-euri-api-key --data-file=-

# 2) Build & push
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/bookforge/bookforge:latest .

# 3) Create the job (generous timeout: whole channels take a while)
gcloud run jobs create bookforge \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/bookforge/bookforge:latest \
  --region $REGION --cpu 2 --memory 4Gi --task-timeout 4h --max-retries 1 \
  --set-secrets EURI_API_KEY=bookforge-euri-api-key:latest \
  --set-env-vars BOOKFORGE_WORKSPACE_ROOT=/data

# 4) Run a channel
gcloud run jobs execute bookforge --region $REGION \
  --update-env-vars CHANNEL_URL=https://www.youtube.com/@TargetChannel
```

The job entrypoint is `python -m bookforge.main`; pass the URL by overriding
args (`--args`) or make the container read `$CHANNEL_URL` via a wrapper script.
Mount a **GCS bucket as a volume** (Cloud Run volume mounts) at `/data` so the
workspace (manifest, chapters, `book/book.pdf`) survives the task.

Scale-to-zero is inherent: jobs bill only while executing.

## Option B — local / VM

```bash
pip install ".[pipeline]"
export EURI_API_KEY=...
bookforge "https://www.youtube.com/@TargetChannel"   # or: python -m bookforge.main <url>
```

## Observability

- The CLI prints one line per agent event; Cloud Run captures stdout/stderr in
  Cloud Logging automatically.
- Each video's checkpoint lives in `data/<channel>/manifest.json` — inspect it
  to see per-video status (`verified / written / failed`) after any run.

## Security notes

- No credentials in the image; Secret Manager only.
- The container runs as a non-root user.
- Process only content you have rights to (own channel / licensed).

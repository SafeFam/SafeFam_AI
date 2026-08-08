#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_ROOT="/opt/safefam"
AI_DIR="${DEPLOY_ROOT}/SafeFam_AI"
BE_DIR="${DEPLOY_ROOT}/SafeFam_BE"
LOCK_FILE="${DEPLOY_ROOT}/.deploy.lock"
CONTAINER_NAME="safefam-ai-server"

echo "[deploy] Waiting for deployment lock"

exec 9>"${LOCK_FILE}"

if ! flock -w 600 9; then
  echo "[deploy] Another deployment is still running"
  exit 1
fi

echo "[deploy] Updating SafeFam_AI develop branch"

cd "${AI_DIR}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[deploy] AI repository has uncommitted changes"
  exit 1
fi

git fetch origin develop
git switch develop
git pull --ff-only origin develop

DEPLOY_SHA="$(git rev-parse HEAD)"
echo "[deploy] Deploying commit: ${DEPLOY_SHA}"

echo "[deploy] Building fastapi-ai"

cd "${BE_DIR}"

docker compose config --quiet
docker compose build fastapi-ai

echo "[deploy] Replacing fastapi-ai only"

docker compose up -d --no-deps fastapi-ai

echo "[deploy] Waiting for health check"

for attempt in $(seq 1 30); do
  health_status="$(
    docker inspect \
      --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
      "${CONTAINER_NAME}" 2>/dev/null || true
  )"

  echo "[deploy] Attempt ${attempt}/30: ${health_status:-not-found}"

  if [ "${health_status}" = "healthy" ]; then
    echo "[deploy] AI deployment succeeded: ${DEPLOY_SHA}"
    docker image prune -f
    docker builder prune -f --filter "until=168h"
    exit 0
  fi

  if [ "${health_status}" = "unhealthy" ]; then
    break
  fi

  sleep 5
done

echo "[deploy] AI health check failed"
docker logs --tail 100 "${CONTAINER_NAME}" || true
exit 1
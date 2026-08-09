#!/usr/bin/env bash

set -Eeuo pipefail

DEPLOY_ROOT="/opt/safefam"
AI_DIR="${DEPLOY_ROOT}/SafeFam_AI"
BE_DIR="${DEPLOY_ROOT}/SafeFam_BE"
LOCK_FILE="${DEPLOY_ROOT}/.deploy.lock"

SERVICE_NAME="fastapi-ai"
CONTAINER_NAME="safefam-ai-server"

EXPECTED_SHA="${1:?Expected Git commit SHA is required}"

echo "[deploy] Waiting for deployment lock"

exec 9>"${LOCK_FILE}"

if ! flock -w 600 9; then
  echo "[deploy] Another BE or AI deployment is still running"
  exit 1
fi

if ! [[ "${EXPECTED_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[deploy] Invalid commit SHA: ${EXPECTED_SHA}"
  exit 1
fi

echo "[deploy] Updating SafeFam_AI develop branch"

cd "${AI_DIR}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[deploy] AI repository has uncommitted changes"
  exit 1
fi

git fetch origin develop:refs/remotes/origin/develop
git switch -C develop --track origin/develop
git merge --ff-only origin/develop

DEPLOY_SHA="$(git rev-parse HEAD)"

if [ "${DEPLOY_SHA}" != "${EXPECTED_SHA}" ]; then
  echo "[deploy] Commit mismatch"
  echo "[deploy] Expected: ${EXPECTED_SHA}"
  echo "[deploy] Actual:   ${DEPLOY_SHA}"
  exit 1
fi

echo "[deploy] Deploying commit: ${DEPLOY_SHA}"

echo "[deploy] Validating Docker Compose configuration"

cd "${BE_DIR}"

docker compose config --quiet

PREVIOUS_IMAGE_ID="$(
  docker inspect \
    --format='{{.Image}}' \
    "${CONTAINER_NAME}" 2>/dev/null || true
)"

PREVIOUS_IMAGE_REF="$(
  docker inspect \
    --format='{{.Config.Image}}' \
    "${CONTAINER_NAME}" 2>/dev/null || true
)"

if [ -n "${PREVIOUS_IMAGE_ID}" ]; then
  echo "[deploy] Previous image: ${PREVIOUS_IMAGE_ID}"
fi

echo "[deploy] Building ${SERVICE_NAME}"

docker compose build "${SERVICE_NAME}"

echo "[deploy] Replacing ${SERVICE_NAME} only"

docker compose up \
  -d \
  --no-deps \
  "${SERVICE_NAME}"

echo "[deploy] Waiting for AI health check"

for attempt in $(seq 1 60); do
  health_status="$(
    docker inspect \
      --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
      "${CONTAINER_NAME}" 2>/dev/null || true
  )"

  echo "[deploy] Health check ${attempt}/60: ${health_status:-not-found}"

  if [ "${health_status}" = "healthy" ] || \
     [ "${health_status}" = "running" ]; then
    echo "[deploy] AI deployment succeeded: ${DEPLOY_SHA}"

    docker image prune -f
    docker builder prune -f --filter "until=168h"

    exit 0
  fi

  if [ "${health_status}" = "unhealthy" ] || \
     [ "${health_status}" = "exited" ] || \
     [ "${health_status}" = "dead" ]; then
    break
  fi

  sleep 5
done

echo "[deploy] AI health check failed"

docker logs --tail 100 "${CONTAINER_NAME}" || true

if [ -n "${PREVIOUS_IMAGE_ID}" ] && [ -n "${PREVIOUS_IMAGE_REF}" ]; then
  echo "[rollback] Restoring previous image: ${PREVIOUS_IMAGE_ID}"

  if docker tag "${PREVIOUS_IMAGE_ID}" "${PREVIOUS_IMAGE_REF}" && \
     docker compose up \
       -d \
       --no-deps \
       --force-recreate \
       "${SERVICE_NAME}"; then
    echo "[rollback] Previous AI image restored"
  else
    echo "[rollback] Failed to restore previous AI image"
  fi
else
  echo "[rollback] No previous AI image is available"
fi

exit 1
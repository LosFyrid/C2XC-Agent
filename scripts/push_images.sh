#!/usr/bin/env bash
set -euo pipefail

# Build + push backend and frontend images to Harbor.
#
# Defaults:
# - backend:  harbor.pic-aichem.online/sunyk/c2xc-agent-backend:v0.1
# - frontend: harbor.pic-aichem.online/sunyk/c2xc-agent-frontend:v0.1
#
# Usage:
#   ./scripts/push_images.sh
#   TAG=v0.2 ./scripts/push_images.sh
#   ./scripts/push_images.sh --tag v0.2
#   ./scripts/push_images.sh --backend-repo harbor.../c2xc-agent-backend --frontend-repo harbor.../c2xc-agent-frontend --tag v0.1
#
# Notes:
# - Requires `docker` and an authenticated `docker login harbor.pic-aichem.online`.
# - Frontend reads API base at build time (Vite); default is `/api/v1`.

TAG="${TAG:-v0.2}"
BACKEND_REPO="${BACKEND_REPO:-harbor.pic-aichem.online/sunyk/c2xc-agent-backend}"
FRONTEND_REPO="${FRONTEND_REPO:-harbor.pic-aichem.online/sunyk/c2xc-agent-frontend}"
VITE_API_BASE_URL="${VITE_API_BASE_URL:-/api/v1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --backend-repo)
      BACKEND_REPO="${2:-}"
      shift 2
      ;;
    --frontend-repo)
      FRONTEND_REPO="${2:-}"
      shift 2
      ;;
    --vite-api-base-url)
      VITE_API_BASE_URL="${2:-}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TAG}" ]]; then
  echo "TAG is empty" >&2
  exit 2
fi
if [[ -z "${BACKEND_REPO}" ]]; then
  echo "BACKEND_REPO is empty" >&2
  exit 2
fi
if [[ -z "${FRONTEND_REPO}" ]]; then
  echo "FRONTEND_REPO is empty" >&2
  exit 2
fi

BACKEND_IMAGE="${BACKEND_REPO}:${TAG}"
FRONTEND_IMAGE="${FRONTEND_REPO}:${TAG}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/4] Build backend: ${BACKEND_IMAGE}"
docker build -t "${BACKEND_IMAGE}" "${ROOT_DIR}"

echo "[2/4] Push backend: ${BACKEND_IMAGE}"
docker push "${BACKEND_IMAGE}"

echo "[3/4] Build frontend: ${FRONTEND_IMAGE} (VITE_API_BASE_URL=${VITE_API_BASE_URL})"
docker build \
  -t "${FRONTEND_IMAGE}" \
  -f "${ROOT_DIR}/frontend/Dockerfile" \
  "${ROOT_DIR}/frontend" \
  --build-arg "VITE_API_BASE_URL=${VITE_API_BASE_URL}"

echo "[4/4] Push frontend: ${FRONTEND_IMAGE}"
docker push "${FRONTEND_IMAGE}"

echo "Done."
echo "  backend : ${BACKEND_IMAGE}"
echo "  frontend: ${FRONTEND_IMAGE}"


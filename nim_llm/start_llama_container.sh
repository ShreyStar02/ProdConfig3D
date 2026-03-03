#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-NIM_LLM}"
PRODCONFIG_NIM__PORT="${PRODCONFIG_NIM__PORT:-19002}"
PRODCONFIG_NIM__MODEL="${PRODCONFIG_NIM__MODEL:-meta/llama-3.1-70b-instruct}"
PRODCONFIG_NIM__REPOSITORY="${PRODCONFIG_NIM__REPOSITORY:-nim/${PRODCONFIG_NIM__MODEL}}"
PRODCONFIG_NIM__TAG="${PRODCONFIG_NIM__TAG:-latest}"
PRODCONFIG_NIM__IMAGE="${PRODCONFIG_NIM__IMAGE:-nvcr.io/${PRODCONFIG_NIM__REPOSITORY}:${PRODCONFIG_NIM__TAG}}"
LOCAL_NIM_CACHE="${PRODCONFIG_NIM__CACHE_DIR:-${SCRIPT_DIR}/.nim-cache}"

masked_prefix() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo "<empty>"
    return
  fi
  echo "${value:0:12}"
}

if [[ -z "${PRODCONFIG_NIM__API_KEY:-}" ]]; then
  echo "ERROR: This is not supported/compatible - missing required environment variable PRODCONFIG_NIM__API_KEY"
  exit 1
fi

# Normalize potential CRLF artifacts from .env edits on Windows.
PRODCONFIG_NIM__API_KEY="$(printf '%s' "${PRODCONFIG_NIM__API_KEY}" | tr -d '\r')"
if [[ -z "${PRODCONFIG_NIM__API_KEY}" ]]; then
  echo "ERROR: This is not supported/compatible - PRODCONFIG_NIM__API_KEY is empty after normalization"
  exit 1
fi
if [[ "${PRODCONFIG_NIM__API_KEY}" == *"nvapi-your"* ]] || [[ "${PRODCONFIG_NIM__API_KEY}" == *"your-key"* ]]; then
  echo "ERROR: This is not supported/compatible - PRODCONFIG_NIM__API_KEY is a placeholder"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: This is not supported/compatible - Docker is not installed or not on PATH"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: This is not supported/compatible - Docker daemon is not running"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: This is not supported/compatible - NVIDIA GPU runtime is not available (nvidia-smi missing)"
  exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: This is not supported/compatible - NVIDIA GPU is not accessible from this environment"
  exit 1
fi

mkdir -p "${LOCAL_NIM_CACHE}"

echo "Using NIM image: ${PRODCONFIG_NIM__IMAGE}"
echo "Resolved repository: ${PRODCONFIG_NIM__REPOSITORY}"
echo "Resolved tag: ${PRODCONFIG_NIM__TAG}"
echo "API key fingerprint: prefix=$(masked_prefix "${PRODCONFIG_NIM__API_KEY}") length=${#PRODCONFIG_NIM__API_KEY}"

if docker manifest inspect "${PRODCONFIG_NIM__IMAGE}" >/dev/null 2>&1; then
  echo "nvcr.io auth/entitlement already valid for ${PRODCONFIG_NIM__IMAGE}; skipping docker login"
else
  echo "Logging into nvcr.io with PRODCONFIG_NIM__API_KEY"
  if ! printf '%s' "${PRODCONFIG_NIM__API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin; then
    echo "ERROR: This is not supported/compatible - docker login failed for nvcr.io"
    exit 1
  fi

  if ! docker manifest inspect "${PRODCONFIG_NIM__IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: This is not supported/compatible - authenticated but no entitlement for ${PRODCONFIG_NIM__IMAGE}."
    exit 1
  fi
fi

if docker image inspect "${PRODCONFIG_NIM__IMAGE}" >/dev/null 2>&1; then
  echo "Image already present locally; skipping docker pull"
elif ! docker pull "${PRODCONFIG_NIM__IMAGE}"; then
  echo "ERROR: This is not supported/compatible - failed to pull image ${PRODCONFIG_NIM__IMAGE}."
  echo "Reason: the repository/tag may not be entitled for this API key."
  echo "Try setting PRODCONFIG_NIM__REPOSITORY and PRODCONFIG_NIM__TAG to an entitled model (for example nim/meta/llama-3.1-70b-instruct:latest)."
  exit 1
fi

echo "Starting container ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --name "${CONTAINER_NAME}" -it --rm \
  --gpus all \
  -p "${PRODCONFIG_NIM__PORT}:${PRODCONFIG_NIM__PORT}" \
  --shm-size=16GB \
  -e NGC_API_KEY="${PRODCONFIG_NIM__API_KEY}" \
  -e HF_HOME=/opt/nim/.cache/huggingface \
  -e TRANSFORMERS_OFFLINE=0 \
  -e NIM_RELAX_MEM_CONSTRAINTS="${PRODCONFIG_NIM__RELAX_MEM_CONSTRAINTS:-1}" \
  -e NIM_SERVED_MODEL_NAME="${PRODCONFIG_NIM__MODEL}" \
  -e NIM_OFFLOADING_POLICY="${PRODCONFIG_NIM__OFFLOADING_POLICY:-system_ram}" \
  -e NIM_TRITON_REQUEST_TIMEOUT="${PRODCONFIG_NIM__TRITON_REQUEST_TIMEOUT:-1800000000}" \
  -e NIM_NUM_GPUS="${PRODCONFIG_NIM__NUM_GPUS:-1}" \
  -e NIM_GPU_MEMORY_FRACTION="${PRODCONFIG_NIM__GPU_MEMORY_FRACTION:-0.85}" \
  -e NIM_HTTP_API_PORT="${PRODCONFIG_NIM__PORT}" \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -u "$(id -u)" \
  "${PRODCONFIG_NIM__IMAGE}"

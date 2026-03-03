# Setup Guide

## Prerequisites

- Python 3.10+
- Bash shell (`bash`) on both Windows and Linux
- Docker Desktop / Docker Engine running
- NVIDIA GPU runtime available (`nvidia-smi` must work)

## Standard Setup (Windows and Linux)

```bash
cd ProdConfig
bash setup.sh
```

`setup.sh` creates/uses `venv` and installs all required Python dependencies from `requirements.txt`.

## Canonical NIM Configuration

Set these in `.env`:

```bash
PRODCONFIG_NIM__API_KEY=nvapi-your-key-here
PRODCONFIG_NIM__PROFILE=local
PRODCONFIG_NIM__AUTH_MODE=none
PRODCONFIG_NIM__PORT=19002
PRODCONFIG_NIM__MODEL=meta/llama-3.1-70b-instruct
```

## Start Local Llama NIM

```bash
python nim_llm/run_llama.py
```

## Verify Runtime

```bash
python main.py nim-smoke --nim-profile local
python main.py step1 --source input/shoe.usd --dest output/shoe_run
```

## Compatibility Policy

The project now fails fast for incompatible environments. Required components are not skipped.

Expected hard error format:

```text
ERROR: This is not supported/compatible - <reason>
```

Common reasons:

- Docker not installed or daemon not running
- NVIDIA runtime missing or inaccessible
- Required Python dependency missing from environment

# Self-Hosted LLM Setup Guide

This guide covers deploying the RAG chatbot with self-hosted inference on an L4 GPU.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      L4 GPU (24GB VRAM)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────┐    │
│  │     vLLM Server     │    │    Embedding Server     │    │
│  │  Llama 3.1 8B AWQ   │    │  (CPU or shared GPU)    │    │
│  │     Port: 8000      │    │      Port: 8080         │    │
│  │                     │    │                         │    │
│  │  - Prefix Caching   │    │  - BGE-small or         │    │
│  │  - Cont. Batching   │    │  - all-MiniLM-L6-v2     │    │
│  │  - AWQ Quantization │    │                         │    │
│  └─────────────────────┘    └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   FastAPI App       │
                    │   Port: 8080        │
                    └─────────────────────┘
```

## Quick Start

### 1. Install vLLM

```bash
# Create virtual environment
python -m venv vllm-env
source vllm-env/bin/activate

# Install vLLM (requires CUDA)
pip install vllm
```

### 2. Start vLLM Server

```bash
# Using the provided script
chmod +x scripts/start_vllm.sh
./scripts/start_vllm.sh

# Or manually:
python -m vllm.entrypoints.openai.api_server \
    --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --quantization awq \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --port 8000
```

### 3. Configure the App

```bash
# Set environment variables
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4

# For embeddings, use local model (no OpenAI needed)
export EMBEDDING_BACKEND=local
export LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Start the app
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 4. Run Benchmarks

```bash
# Benchmark vLLM
python scripts/benchmark_inference.py --backend vllm --runs 10

# Compare with OpenAI (if you have a key)
python scripts/benchmark_inference.py --backend openai --runs 10
```

## Model Options for L4 (24GB)

| Model | VRAM | Quality | Speed | Command |
|-------|------|---------|-------|---------|
| Llama 3.1 8B AWQ | ~5GB | Good | Fast | `--model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` |
| Qwen 2.5 7B AWQ | ~4GB | Good | Fast | `--model Qwen/Qwen2.5-7B-Instruct-AWQ` |
| Mistral 7B AWQ | ~4GB | Good | Very Fast | `--model TheBloke/Mistral-7B-Instruct-v0.2-AWQ` |
| Llama 3.1 8B FP16 | ~16GB | Best | Medium | `--model meta-llama/Llama-3.1-8B-Instruct` (no quantization flag) |

## Optimization Flags

### Prefix Caching
Caches KV states for repeated prefixes (system prompt).

```bash
--enable-prefix-caching
```

**Effect:** 2-10x speedup for cached prefixes (your system prompt).

### Chunked Prefill
Splits long prefills to interleave with decodes.

```bash
--enable-chunked-prefill
```

**Effect:** Better latency for concurrent users.

### GPU Memory Utilization
How much VRAM to use (leave headroom for KV cache).

```bash
--gpu-memory-utilization 0.90  # Use 90% of VRAM
```

## Expected Performance on L4

| Metric | Value |
|--------|-------|
| TTFT (Time to First Token) | 100-200ms |
| Decode Speed | 40-60 tokens/sec |
| Max Concurrent Users | 8-12 |
| Max Context Length | 8192 tokens |

## Troubleshooting

### Out of Memory
```bash
# Reduce context length
--max-model-len 4096

# Or reduce GPU utilization
--gpu-memory-utilization 0.85
```

### Slow First Request
First request downloads and loads the model. Subsequent requests are fast.

### Model Not Found
Make sure you have access to gated models:
```bash
huggingface-cli login
```

## Monitoring

vLLM exposes metrics at `/metrics` (Prometheus format):

```bash
curl http://localhost:8000/metrics
```

Key metrics:
- `vllm:num_requests_running` - Active requests
- `vllm:num_requests_waiting` - Queued requests  
- `vllm:gpu_cache_usage_perc` - KV cache utilization
- `vllm:avg_generation_throughput_toks_per_s` - Tokens/sec


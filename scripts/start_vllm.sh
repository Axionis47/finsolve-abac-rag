#!/bin/bash
# =============================================================================
# vLLM Server Startup Script for L4 GPU (24GB)
# =============================================================================
#
# This script starts vLLM with optimized settings for:
# - Llama 3.1 8B Instruct (AWQ quantized)
# - L4 GPU with 24GB VRAM
# - Prefix caching enabled
# - Continuous batching
#
# Usage:
#   ./scripts/start_vllm.sh                    # Default settings
#   ./scripts/start_vllm.sh --model <model>    # Custom model
#   ./scripts/start_vllm.sh --benchmark        # Run with benchmarking enabled
# =============================================================================

set -e

# Default configuration
MODEL="${VLLM_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTIL:-0.90}"

# Optimization flags
ENABLE_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-true}"
ENABLE_CHUNKED_PREFILL="${VLLM_CHUNKED_PREFILL:-true}"
QUANTIZATION="${VLLM_QUANTIZATION:-awq}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK_MODE=true
            shift
            ;;
        --no-prefix-cache)
            ENABLE_PREFIX_CACHING=false
            shift
            ;;
        --fp16)
            QUANTIZATION=""
            MODEL="meta-llama/Llama-3.1-8B-Instruct"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================="
echo "vLLM Server Configuration"
echo "============================================="
echo "Model:              $MODEL"
echo "Port:               $PORT"
echo "Max Context Length: $MAX_MODEL_LEN"
echo "GPU Memory Util:    $GPU_MEMORY_UTILIZATION"
echo "Prefix Caching:     $ENABLE_PREFIX_CACHING"
echo "Chunked Prefill:    $ENABLE_CHUNKED_PREFILL"
echo "Quantization:       ${QUANTIZATION:-none}"
echo "============================================="

# Build vLLM command
CMD="python -m vllm.entrypoints.openai.api_server"
CMD="$CMD --model $MODEL"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --max-model-len $MAX_MODEL_LEN"
CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

# Add optimization flags
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
    CMD="$CMD --enable-prefix-caching"
fi

if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    CMD="$CMD --enable-chunked-prefill"
fi

if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

# Tensor parallelism (for multi-GPU, not needed for single L4)
# CMD="$CMD --tensor-parallel-size 1"

# Trust remote code (needed for some models)
CMD="$CMD --trust-remote-code"

# API key (optional, for security)
if [ -n "$VLLM_API_KEY" ]; then
    CMD="$CMD --api-key $VLLM_API_KEY"
fi

echo ""
echo "Starting vLLM server..."
echo "Command: $CMD"
echo ""

# Run vLLM
exec $CMD



import modal

# ---------------------------------------------------------------------------
# Configuration — toggle these for debugging / production
# ---------------------------------------------------------------------------

# --- Production config (MiniMax-M2.5 on 4x B200) ---
# MODEL_NAME = "MiniMaxAI/MiniMax-M2.5"
# MODEL_REVISION = "main"
# N_GPU = 4
# GPU_CONFIG = f"B200:{N_GPU}"  # string format per Modal docs: "B200:4"

# --- Debug config (uncomment the block below and comment out the block above
#     to test the pipeline with a smaller model on a single GPU) ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "main"
N_GPU = 1
GPU_CONFIG = f"H100:{N_GPU}"  # single GPU, no count suffix needed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MINUTES = 60  # seconds
VLLM_PORT = 8000

# ---------------------------------------------------------------------------
# Modal Volumes — persistent caches shared across container restarts
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("uv")
    .run_commands(
        # vLLM nightly via uv — stable pip resolves to 0.15.1 which
        # doesn't support Qwen3.5 (qwen3_5_moe architecture).
        "uv pip install vllm --prerelease=allow --extra-index-url https://wheels.vllm.ai/nightly --system",
        "uv pip install fastapi[standard] uvicorn[standard] --system",
    )
)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App("minimax-vllm-openai-server")


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    # Keep the container warm for 5 minutes after the last request to save costs.
    scaledown_window=5 * MINUTES,
    # Allow up to 10 minutes for the container to start (model download + load).
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """
    Spawns a vLLM OpenAI-compatible server as a subprocess.

    Modal's @modal.web_server decorator will route incoming HTTP traffic to
    the specified port once the server is ready. vLLM natively serves the
    full OpenAI-compatible API surface including:
        - POST /v1/chat/completions  (with streaming support)
        - POST /v1/completions
        - GET  /v1/models
    """
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level", "info",
        # Tensor parallelism — must match the GPU count.
        "--tensor-parallel-size", str(N_GPU),
        # Required for models with custom code (e.g. MiniMax-M2.5).
        "--trust-remote-code",
        # Disable CUDA graph capture for faster cold starts.
        # Remove this flag (or use --no-enforce-eager) for higher throughput
        # once you've verified everything works.
        "--enforce-eager",
    ]

    # Qwen3+ models need reasoning and tool-call parsers; Qwen2.5 does not.
    if MODEL_NAME.startswith("Qwen/Qwen3"):
        cmd += [
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
        ]

    print(f"[modal_vllm_server] Starting vLLM with command:\n  {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)

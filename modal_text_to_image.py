
# ---------------------------------------------------------------------------
# Text-to-Image Server on Modal (GLM-Image)
# ---------------------------------------------------------------------------
#
# Deploy:
#   modal deploy modal_text_to_image.py
#
# Generate image (returns URL by default):
#   curl -X POST "https://zhongyi070622--text-to-image-server-inference-generate-dev.modal.run" \
#     -H "Content-Type: application/json" \
#     -d '{
#       "prompt": "A cat sitting on a rainbow",
#       "n": 1,
#       "size": "1024x1024",
#       "response_format": "url"
#     }'
#
# Generate image (returns base64):
#   curl -X POST "https://zhongyi070622--text-to-image-server-inference-generate-dev.modal.run" \
#     -H "Content-Type: application/json" \
#     -d '{
#       "prompt": "A cat sitting on a rainbow",
#       "n": 1,
#       "size": "1024x1024",
#       "response_format": "b64_json"
#     }'
#
# Fetch a generated image by ID:
#   curl "https://zhongyi070622--text-to-image-server-inference-images-dev.modal.run?image_id=<id>" \
#     --output image.png
#
# Quick generate (returns PNG directly):
#   curl "https://zhongyi070622--text-to-image-server-inference-web-dev.modal.run?prompt=A+cat+sitting+on+a+rainbow" \
#     --output test.png
#
# Health check:
#   curl "https://zhongyi070622--text-to-image-server-inference-health-dev.modal.run"
#
# ---------------------------------------------------------------------------

import modal
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "zai-org/GLM-Image"
PIPELINE_CLASS = "GlmImagePipeline"
N_GPU = 1
GPU_CONFIG = f"B200:{N_GPU}"
DEFAULT_NUM_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 1.5
DEFAULT_WIDTH = 960   # must be divisible by 32
DEFAULT_HEIGHT = 1280  # must be divisible by 32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MINUTES = 60  # seconds
CACHE_DIR = "/cache"
IMAGES_DIR = "/generated-images"
IMAGES_ENDPOINT_URL = "https://zhongyi070622--text-to-image-server-inference-images-dev.modal.run"

# ---------------------------------------------------------------------------
# Modal Volumes
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
images_vol = modal.Volume.from_name("generated-images", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container image — use NVIDIA CUDA base for B200 (sm_100) compatibility
# ---------------------------------------------------------------------------
cuda_version = "12.8.1"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1")
    .uv_pip_install(
        "accelerate==1.2.1",
        "fastapi[standard]==0.115.4",
        "sentencepiece==0.2.0",
        "git+https://github.com/huggingface/transformers.git",
        "git+https://github.com/huggingface/diffusers.git",
    )
    .uv_pip_install(
        "torch==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

with image.imports():
    import diffusers
    import torch
    from fastapi import Response, HTTPException

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App("text-to-image-server")


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    volumes={CACHE_DIR: hf_cache_vol, IMAGES_DIR: images_vol},
)
class Inference:
    @modal.enter()
    def load_pipeline(self):
        """Load the diffusion pipeline onto the GPU once at container start."""
        print(f"[text_to_image] Loading {MODEL_ID} with {PIPELINE_CLASS}...")

        pipeline_cls = getattr(diffusers, PIPELINE_CLASS)
        self.pipe = pipeline_cls.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        print("[text_to_image] Pipeline loaded on GPU.")

    @modal.method()
    def run(
        self,
        prompt: str,
        batch_size: int = 1,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        num_inference_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: Optional[int] = None,
    ) -> list[bytes]:
        """Generate images and return them as PNG bytes."""
        import io
        import random

        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print(f"seeding RNG with {seed}")
        torch.manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "num_images_per_prompt": batch_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
        }

        images = self.pipe(**kwargs).images

        image_output = []
        for img in images:
            with io.BytesIO() as buf:
                img.save(buf, format="PNG")
                image_output.append(buf.getvalue())

        torch.cuda.empty_cache()
        return image_output

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict):
        """
        OpenAI-compatible image generation endpoint.

        POST /generate
        {
            "prompt": "A cat sitting on a rainbow",
            "n": 1,
            "size": "512x512",
            "response_format": "url"   // "url" (default) or "b64_json"
        }
        """
        import base64
        import os
        import time
        import uuid

        prompt = request.get("prompt", "")
        n = request.get("n", 1)
        size = request.get("size", f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
        response_format = request.get("response_format", "url")
        num_inference_steps = request.get("num_inference_steps", DEFAULT_NUM_STEPS)
        guidance_scale = request.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        seed = request.get("seed", None)

        # Parse "WxH" size string
        try:
            w, h = size.lower().split("x")
            width, height = int(w), int(h)
        except (ValueError, AttributeError):
            width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT

        image_bytes_list = self.run.local(
            prompt=prompt,
            batch_size=n,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        data = []
        for img_bytes in image_bytes_list:
            if response_format == "b64_json":
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                data.append({"b64_json": b64})
            else:
                # Save to Volume and return URL
                image_id = uuid.uuid4().hex
                os.makedirs(IMAGES_DIR, exist_ok=True)
                filepath = os.path.join(IMAGES_DIR, f"{image_id}.png")
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
                images_vol.commit()

                url = f"{IMAGES_ENDPOINT_URL}?image_id={image_id}"
                data.append({"url": url})

        return {
            "created": int(time.time()),
            "data": data,
        }

    @modal.fastapi_endpoint(docs=True)
    def images(self, image_id: str):
        """Serve a previously generated image by its ID."""
        import os

        filepath = os.path.join(IMAGES_DIR, f"{image_id}.png")
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Image not found")

        with open(filepath, "rb") as f:
            content = f.read()

        return Response(content=content, media_type="image/png")

    @modal.fastapi_endpoint(docs=True)
    def web(self, prompt: str, seed: Optional[int] = None):
        """Simple endpoint that returns a PNG image directly."""
        return Response(
            content=self.run.local(prompt, batch_size=1, seed=seed)[0],
            media_type="image/png",
        )

    @modal.fastapi_endpoint(docs=True)
    def health(self):
        """Health check endpoint."""
        return {
            "status": "ok",
            "model": MODEL_ID,
            "gpu": GPU_CONFIG,
            "pipeline": PIPELINE_CLASS,
        }

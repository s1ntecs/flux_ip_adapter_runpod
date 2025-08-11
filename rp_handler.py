# -*- coding: utf-8 -*-
import base64, io, random, time, inspect
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from diffusers import FluxImg2ImgPipeline

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --------------------------- КОНСТАНТЫ ----------------------------------- #
MAX_SEED = np.iinfo(np.int32).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 250
TARGET_RES = 1024

logger = RunPodLogger()


# ------------------------- УТИЛИТЫ --------------------------------------- #
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def round_to_multiple(x, m=8):
    return (x // m) * m


def compute_work_resolution(w, h, max_side=1024):
    # масштабируем так, чтобы большая сторона <= max_side, и кратность 8
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(int(w * scale), 8)
    new_h = max(int(h * scale), 8)
    new_w = max(round_to_multiple(new_w, 8), 8)
    new_h = max(round_to_multiple(new_h, 8), 8)
    return new_w, new_h


def call_with_supported_kwargs(fn, **kwargs):
    """Фильтруем kwargs по сигнатуре fn (версии diffusers бывают разные)."""
    sig = inspect.signature(fn)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**allowed)


# ------------------------- ЗАГРУЗКА МОДЕЛИ ------------------------------- #
logger.info("LOADING FluxImg2ImgPipeline + IP-Adapter")

BASE_REPO = "black-forest-labs/FLUX.1-dev"
PIPELINE = FluxImg2ImgPipeline.from_pretrained(
    BASE_REPO,
    torch_dtype=DTYPE
).to(DEVICE)

# IP-Adapter для FLUX
ADAPTER_NAME = "XLabs-AI/flux-ip-adapter"
ENCODER_REPO = "openai/clip-vit-large-patch14"

PIPELINE.load_ip_adapter(
    ADAPTER_NAME,
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path=ENCODER_REPO
)
logger.info("IP-Adapter READY")


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})

        # обязательные поля
        image_url = payload.get("image_url")
        style_url = payload.get("style_image_url")
        prompt = payload.get("prompt")

        if not image_url or not style_url:
            return {"error": "'image_url' и 'style_image_url' обязательны"}
        if not prompt:
            return {"error": "'prompt' обязателен"}

        # необязательные
        neg_prompt = payload.get("neg_prompt")
        steps = min(int(payload.get("steps", 28)), MAX_STEPS)
        seed = int(payload.get("seed", random.randint(0, MAX_SEED)))
        guidance_scale = float(payload.get("guidance_scale", 3.5))
        true_cfg_scale = float(payload.get("true_cfg_scale", 4.0))
        strength = float(payload.get("strength", 0.55))
        ip_adapter_scale = float(payload.get("ip_adapter_scale", 1.0))

        # загрузка изображений
        init_pil = url_to_pil(image_url)
        w0, h0 = init_pil.size
        work_w, work_h = compute_work_resolution(w0, h0, TARGET_RES)
        init_pil = init_pil.resize((work_w, work_h), Image.Resampling.LANCZOS)

        ref_pil = url_to_pil(style_url).resize((work_w, work_h),
                                               Image.Resampling.LANCZOS)

        # сид
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # IP-Adapter scale на прогона
        PIPELINE.set_ip_adapter_scale(ip_adapter_scale)

        # вызов пайплайна (через фильтр аргументов)
        out = call_with_supported_kwargs(
            PIPELINE,
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=init_pil,
            ip_adapter_image=ref_pil,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
            width=work_w, height=work_h
        )
        images = out.images

        return {
            "images_base64": [pil_to_b64(i) for i in images],
            "time": round(time.time() - job["created"], 2) if "created" in job else None,
            "steps": steps,
            "seed": seed,
            "width": work_w,
            "height": work_h
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps', 'strength' или размер изображения."}
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

# import cv2
import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict
from PIL import Image

from diffusers import FluxPipeline

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


# ------------------------- ФУНКЦИИ-ПОМОЩНИКИ ----------------------------- #
def filter_items(colors_list, items_list, items_to_remove):
    keep_c, keep_i = [], []
    for c, it in zip(colors_list, items_list):
        if it not in items_to_remove:
            keep_c.append(c)
            keep_i.append(it)
    return keep_c, keep_i


def resize_dimensions(dimensions, target_size):
    w, h = dimensions
    if w < target_size and h < target_size:
        return dimensions
    if w > h:
        ar = h / w
        return target_size, int(target_size * ar)
    ar = w / h
    return int(target_size * ar), target_size


def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def round_to_multiple(x, m=8):
    return (x // m) * m


def compute_work_resolution(w, h, max_side=1024):
    # масштабируем так, чтобы большая сторона <= max_side
    scale = min(max_side / max(w, h), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # выравниваем до кратных 8
    new_w = round_to_multiple(new_w, 8)
    new_h = round_to_multiple(new_h, 8)
    return max(new_w, 8), max(new_h, 8)


def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #

logger.info("controlnet IS READY")
repo_id = "black-forest-labs/FLUX.1-dev"
PIPELINE = FluxPipeline.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16
).to(DEVICE)


logger.info("PIPELINE IS READY")

PIPELINE.load_ip_adapter(
    "XLabs-AI/flux-ip-adapter",
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
)

logger.info("IP_ADAPTER IS READY")


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        logger.info("HANDLER START")

        payload = job.get("input", {})
        image_url = payload.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required"}

        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        neg_prompt = payload.get("neg_prompt")

        steps = min(int(payload.get(
            "steps", MAX_STEPS)),
                    MAX_STEPS)

        seed = int(payload.get(
            "seed",
            random.randint(0, MAX_SEED)))

        guidance_scale = float(payload.get(
            "guidance_scale", 3.5))

        true_cfg_scale = float(payload.get(
            "true_cfg_scale", 4))

        ip_adapter_scale = float(payload.get(
            "ip_adapter_scale", 1.0))

        generator = torch.Generator(
            device=DEVICE).manual_seed(seed)

        image_pil = url_to_pil(image_url)

        orig_w, orig_h = image_pil.size
        work_w, work_h = compute_work_resolution(orig_w, orig_h, TARGET_RES)

        image_pil = image_pil.resize((work_w, work_h),
                                     Image.Resampling.LANCZOS)

        PIPELINE.set_ip_adapter_scale(ip_adapter_scale)

        # ------------------ генерация ---------------- #
        images = PIPELINE(
            width=work_w,
            height=work_h,
            prompt="wearing sunglasses",
            negative_prompt=neg_prompt,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
            ip_adapter_image=image_pil,
            num_inference_steps=steps
        ).images

        return {
            "images_base64": [pil_to_b64(i) for i in images],
            "time": round(time.time() - job["created"],
                          2) if "created" in job else None,
            "steps": steps, "seed": seed
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."} # noqa
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

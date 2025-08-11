import os
import torch


from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel
from image_gen_aux import DepthPreprocessor

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ------------------------- пайплайн -------------------------
def get_pipeline():
    controlnet = FluxControlNetModel.from_pretrained(
        "InstantX/FLUX.1-dev-Controlnet-Canny",
        torch_dtype=torch.bfloat16
    )

    repo_id = "black-forest-labs/FLUX.1-dev"
    FluxControlNetImg2ImgPipeline.from_pretrained(
        repo_id,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    )

if __name__ == "__main__":
    get_pipeline()

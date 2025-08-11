import os
import torch


from diffusers import FluxImg2ImgPipeline

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- пайплайн -------------------------
def get_pipeline():
    BASE_REPO = "black-forest-labs/FLUX.1-dev"
    PIPELINE = FluxImg2ImgPipeline.from_pretrained(
        BASE_REPO,
        torch_dtype=torch.bfloat16
    )

    # IP-Adapter для FLUX
    ADAPTER_NAME = "XLabs-AI/flux-ip-adapter"
    ENCODER_REPO = "openai/clip-vit-large-patch14"

    PIPELINE.load_ip_adapter(
        ADAPTER_NAME,
        weight_name="ip_adapter.safetensors",
        image_encoder_pretrained_model_name_or_path=ENCODER_REPO
    )


if __name__ == "__main__":
    get_pipeline()

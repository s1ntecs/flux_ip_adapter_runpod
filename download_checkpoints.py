import os
import torch


from diffusers import FluxControlNetPipeline,FluxControlNetModel

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- пайплайн -------------------------
def get_pipeline():
    print("LOADING ControlNet (Canny) and Flux.1-dev")
    CONTROLNET_REPO = "InstantX/FLUX.1-dev-Controlnet-Canny"

    controlnet = FluxControlNetModel.from_pretrained(
        CONTROLNET_REPO, torch_dtype=torch.bfloat16
    )
    print("controlnet IS READY")

    BASE_REPO = "black-forest-labs/FLUX.1-dev"
    PIPELINE = FluxControlNetPipeline.from_pretrained(
        BASE_REPO,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    ADAPTER_NAME = "XLabs-AI/flux-ip-adapter"
    ENCODER_REPO = "openai/clip-vit-large-patch14"

    PIPELINE.load_ip_adapter(
        ADAPTER_NAME,
        weight_name="ip_adapter.safetensors",
        image_encoder_pretrained_model_name_or_path=ENCODER_REPO
    )


if __name__ == "__main__":
    get_pipeline()

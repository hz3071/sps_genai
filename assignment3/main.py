import io
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from torchvision.utils import save_image
from helper_lib.model import Generator

app = FastAPI()

device = torch.device("cpu")

generator = Generator().to(device)
generator.load_state_dict(
    torch.load("generator.pth", map_location=device, weights_only=True)
)
generator.eval()


@app.get("/")
def home():
    return {"message": "MNIST GAN API"}


@app.get("/generate")
def generate():
    noise = torch.randn(1, 100)

    with torch.no_grad():
        image = generator(noise)

    buffer = io.BytesIO()
    save_image(image, buffer, format="png", normalize=True)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
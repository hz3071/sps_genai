import torch
from torchvision.utils import save_image
from gan_model import Generator

device = torch.device("mps")

generator = Generator().to(device)
generator.load_state_dict(
    torch.load("generator.pth", map_location=device, weights_only=True)
)

generator.eval()

noise = torch.randn(16, 100).to(device)

with torch.no_grad():
    images = generator(noise)

save_image(images, "generated_digits.png", nrow=4, normalize=True)

print("Image saved")
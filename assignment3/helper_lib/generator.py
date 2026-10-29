import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, 100, device=device)
        generated_images = model(noise)

    # Convert images from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2

    grid = make_grid(generated_images, nrow=5)

    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.show()

    return generated_images
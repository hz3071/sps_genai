import torch


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    generator, discriminator = model
    optimizer_g, optimizer_d = optimizer

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_d.zero_grad()

            real_outputs = discriminator(real_images)
            loss_real = criterion(real_outputs, real_labels)

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)

            fake_outputs = discriminator(fake_images.detach())
            loss_fake = criterion(fake_outputs, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            noise = torch.randn(batch_size, 100, device=device)
            generated_images = generator(noise)

            outputs = discriminator(generated_images)
            loss_g = criterion(outputs, real_labels)

            loss_g.backward()
            optimizer_g.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"D Loss: {loss_d.item():.4f}, "
            f"G Loss: {loss_g.item():.4f}"
        )

    return generator, discriminator
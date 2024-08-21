import sys
import torch
import os
import numpy as np

from utils import get_all_data_loaders, get_config, prepare_sub_folder

sys.path.append("C:\\_Uni\\Thesis\\Master Project")
from master_utils import normalize, denormalize, find_best_k, plot_latent_space

CHECKPOINT = 'gen_01000000.pt'
SAVE_PATH = 'latent_space.png'
BEST_K = 3

def plot_latent(model, data_loader, num_samples=1000, save_path=SAVE_PATH):
    model.eval()
    latents = []
    # labels = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_samples:
                break
            data = data.cuda()
            h, _ = model.gen_a.encode(data)
            h_flat = h.view(h.size(0), -1)  # Flatten the latent representation
            #print(h_flat.shape())
            latents.append(h_flat.cpu().numpy())
            # labels.extend([i] * h_flat.size(0))  # Create dummy labels for each sample

    latents = np.concatenate(latents, axis=0)

    plot_latent_space(latents, BEST_K, save_path)


if __name__ == "__main__":
    import argparse
    from trainer import UNIT_Trainer
    import torch.backends.cudnn as cudnn
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/unit_LINAC_folder.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
    opts = parser.parse_args()

    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path

    # Setup model and data loader
    if opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support UNIT")

    trainer.cuda()
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

    # Ensure display_size does not exceed the length of the dataset
    display_size = min(display_size, len(train_loader_a.dataset), len(train_loader_b.dataset),
                       len(test_loader_a.dataset), len(test_loader_b.dataset))

    # Modify here to only stack the image part of the dataset item
    train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in range(display_size)]).cuda()

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

    # Resume from last checkpoint
    last_checkpoint = os.path.join(checkpoint_directory, CHECKPOINT)
    if os.path.exists(last_checkpoint):
        iterations = trainer.resume(checkpoint_directory, hyperparameters=config)
        print(f"Resumed from iteration {iterations}")
    else:
        iterations = 0
        print("Starting from scratch")

    # Plot and save latent space
    plot_latent(trainer, train_loader_a, num_samples=1000, save_path=SAVE_PATH)


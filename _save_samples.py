import sys
import torch
import os
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights

# sys.path.append("/mnt/data/Olga/Master Project")
sys.path.append("C:\\_Uni\\Thesis\\Master Project")
from master_utils import normalize, denormalize, plot_individual_metrics, plot_general_metrics, extract_fid_features, calculate_fid, print_and_save_metrics

CHECKPOINT = 'gen_01000000.pt'
SAMPLES_FOLDER = 'generated_samples'

def generate_and_save_samples(model, data_loader_a, data_loader_b, num_samples=100, save_dir='generated_samples'):
    print("Generating and saving samples...")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Initialize a counter for the images
    image_count = 0

    target_images = []
    output_images = []
    input_images = []

    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    inception_model.fc = torch.nn.Identity()

    with torch.no_grad():
        for i, (data_a, data_b) in enumerate(zip(data_loader_a, data_loader_b)):
            if i >= num_samples:
                break

            x_a, x_b = data_a.cuda(), data_b.cuda()

            # Generate samples
            x_ab, x_ba = model(x_a, x_b)

            # Denormalize images to the range [0, 688] for plotting
            x_a = denormalize(x_a)
            x_ab = denormalize(x_ab)
            x_b = denormalize(x_b)
            # x_ba = denormalize(x_ba)

            # Ensure tensors have the correct shape
            if len(x_a.shape) == 4:  # (batch_size, channels, height, width)
                x_a = x_a.cpu().squeeze(0).permute(1, 2, 0).numpy()
                x_ab = x_ab.cpu().squeeze(0).permute(1, 2, 0).numpy()
                x_b = x_b.cpu().squeeze(0).permute(1, 2, 0).numpy()
            # x_ba = x_ba.cpu().squeeze(0).permute(1, 2, 0).numpy()
            elif len(x_a.shape) == 3:  # (channels, height, width)
                x_a = x_a.cpu().permute(1, 2, 0).numpy()
                x_ab = x_ab.cpu().permute(1, 2, 0).numpy()
                x_b = x_b.cpu().permute(1, 2, 0).numpy()
            # x_ba = x_ba.cpu().permute(1, 2, 0).numpy()
            else:
                raise ValueError(f"Unexpected tensor shape: {x_a.shape}")

            # Convert single-channel images to three-channel by duplicating the channel
            target_images.append(np.repeat(x_b, 3, axis=2))
            output_images.append(np.repeat(x_ab, 3, axis=2))
            input_images.append(np.repeat(x_a, 3, axis=2))

            # Calculate and plot metrics
            plot_individual_metrics(image_count, x_a, x_ab, x_b)

            image_count += 1
        #print(f"Samples counter: {image_count}")

    plot_general_metrics()

    # Calculate FID
    target_features = extract_fid_features(inception_model, target_images, device='cuda')
    output_features = extract_fid_features(inception_model, output_images, device='cuda')
    input_features = extract_fid_features(inception_model, input_images, device='cuda')
    fid_it = calculate_fid(input_features, target_features)
    fid_ot = calculate_fid(output_features, target_features)
    fid_oi = calculate_fid(output_features, input_features)
    print(f'FID score: {fid_it}, {fid_ot}, {fid_oi}')

    print_and_save_metrics(fid_it, fid_ot, fid_oi)

    model.train()


if __name__ == "__main__":
    import argparse
    from trainer import UNIT_Trainer
    import torch.backends.cudnn as cudnn
    import shutil
    from utils import get_all_data_loaders, prepare_sub_folder, get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/unit_INPUTDATA_folder.yaml',
                        help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
    opts = parser.parse_args()

    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)
    config['vgg_model_path'] = opts.output_path

    # Setup model and data loader
    if opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support UNIT")

    trainer.cuda()
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

    # Ensure display_size does not exceed the length of the dataset
    display_size = min(len(train_loader_a.dataset), len(train_loader_b.dataset), len(test_loader_a.dataset), len(test_loader_b.dataset))

    # Modify here to only stack the image part of the dataset item
    train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in range(display_size)]).cuda()

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    output_directory = os.path.join(opts.output_path + "\\outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_directory, CHECKPOINT)
    if os.path.exists(checkpoint_path):
        trainer.resume(checkpoint_directory, hyperparameters=config)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        sys.exit(f"Checkpoint {checkpoint_path} not found.")

    # Generate and save samples
    generate_and_save_samples(trainer, test_loader_a, test_loader_b, num_samples=100, save_dir=os.path.join(output_directory, 'generated_samples'))

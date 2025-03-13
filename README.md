Autoencoder Training for Medical Image Reconstruction

This repository contains an implementation of an Autoencoder-based Latent Diffusion Model for medical image reconstruction. The model is trained on 2D slices extracted from medical imaging datasets such as BraTS.

📌 Features

Autoencoder with Variational KL Regularization

Perceptual Loss & Adversarial Training Support

Customizable L1/L2 Loss for Reconstruction

Automatic Model Checkpointing with Best Loss Tracking

TensorBoard Logging for Monitoring Training Progress

🚀 Setup

1️⃣ Clone the repository

git clone https://github.com/yourusername/autoencoder-medical.git
cd autoencoder-medical

2️⃣ Install dependencies

Ensure you have Python 3.8+ and install required libraries:

pip install -r requirements.txt

3️⃣ Prepare the dataset

Place your BraTS dataset or any medical image dataset in the dataset/ folder. The structure should be:

dataset/
  ├── Task01_BrainTumour/
  │   ├── imagesTr/
  │   ├── labelsTr/
  │   ├── dataset.json

Ensure that dataset.json is properly formatted to reference available images.

🏋️‍♂️ Training the Autoencoder

Run the following command to train the Autoencoder:

python train_autoencoder.py --config config/config_train.json

Training Configuration (config/config_train.json):

"autoencoder_train": {
    "batch_size": 1,
    "patch_size": [240, 240],
    "lr": 2.5e-5,
    "perceptual_weight": 2.0,
    "kl_weight": 1e-6,
    "recon_loss": "l2",
    "max_epochs": 1000,
    "val_interval": 1
}

📊 Monitoring Training

TensorBoard Visualization

To monitor training progress, launch TensorBoard:

tensorboard --logdir=logs/

Open http://localhost:6006/ in your browser to visualize:

Loss curves (val_recon_loss)

Reconstructed images (val_recon vs. val_img)

📌 Model Saving Strategy

✅ Save Model Checkpoints

The latest model is always saved as:

trained_weights/diffusion_2d/autoencoder_last.pt
trained_weights/diffusion_2d/discriminator_last.pt

The best model (lowest loss) is saved as:

trained_weights/diffusion_2d/autoencoder.pt
trained_weights/diffusion_2d/discriminator.pt

✅ Save Best Reconstruction Images

Reconstructed images are saved only when validation loss improves:

reconstruction_results/
  ├── best_epoch_12.png
  ├── best_epoch_34.png

🏁 Running Inference

After training, you can generate reconstructions with:

python inference.py --config config/config_infer.json

This will load autoencoder.pt and generate outputs.

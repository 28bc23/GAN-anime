# GAN-Anime

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Pytorch](https://img.shields.io/badge/Pytorch-2.8-green)
![CUDA](https://img.shields.io/badge/CUDA-12.9-green)
![License](https://img.shields.io/badge/License-GPL--3.0-red)

> [!Warning]
> This repository is no longer under active development

## 📄 Description

**GAN-Anime** is a collection of Generative Adversarial Networks (GANs) designed primarily for generating **full-body anime characters** (focusing on female characters). 

This repository allows you to:
* Generate new unique characters using pre-trained models.
* Train your own GAN models on custom datasets.

## 🛠️ Installation

### Prerequisites
* [Anaconda](https://www.anaconda.com/download) or Miniconda
* [CUDA 12.9](https://developer.nvidia.com/cuda-downloads) (Ensure your GPU drivers are compatible)

### Setup Guide

1.  **Create and activate a Conda environment:**
    ```bash
    conda create --name mygan python=3.9
    conda activate mygan
    ```

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/28bc23/GAN-anime.git
    cd GAN-anime
    ```

3.  **Install Dependencies:**
    You can install PyTorch and other requirements using the commands below.
    
    *Using pip (Recommended):*
    ```bash
    # Install PyTorch with CUDA support
    pip3 install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129
    # Note: Adjust 'cu121' to match your specific CUDA version if needed.

    # Install other required libraries
    pip3 install -r requirements.txt
    # if -r requirements.txt doesn't work use this
    pip3 install tensorboard==2.20.0 numpy==1.26.3 matplotlib==3.9.4 Pillow==11.0.0
    ```

## 📂 Dataset Preparation

To train the model, you need a dataset. The default configuration uses the **FBAnimeHQ** dataset.

1.  **Download the dataset:**
    [HuggingFace: skytnt/fbanimehq](https://huggingface.co/datasets/skytnt/fbanimehq)

2.  **Organize the folders:**
    Extract the dataset into a folder named `data` in the root directory. The structure must be as follows:

    ```text
    GAN-anime/
    ├── data/
    │   ├── 0000/
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── 0001/
    │   └── ...
    ├── gan.py
    └── ...
    ```

> **Note:** If you use a custom dataset, ensure it follows this directory structure (numbered folders containing images), or modify the data loading section.

## 🚀 Usage


> [!IMPORTANT]  
> Currently, the gan.py script is under development and is not functional. Therefore, run the given model script directly using for example ``python3 GANs/ProGAN.py``.
> All models are in ``GANs`` directory.

### 1. Generating Images (Inference)

To generate images using the provided pre-trained models:

1.  Choose a model from the `pre-trainedModels` directory.
2.  Copy the model files (e.g., `.pth` files) into the `GANs` folder. (only when using smallGan.py)
    ```bash
    # Example: Copying specific pre-trained weights
    cp pre-trainedModels/small/GAN/5160steps/* ./GANs/
    ```
3.  Run the main script:
    ```bash
    python3 gan.py
    ```

### 2. Training from Scratch

Once your dataset is set up in the `data/` folder:

1.  Open `gan.py` and adjust configurations if necessary (e.g., batch size, image size).
2.  Start the training loop:
    ```bash
    python3 gan.py
    ```
    *The script will automatically look for images in the `data/` directory and begin training.*

## 📜 License

This project is licensed under the **GPL-3.0 License**. See the [LICENSE](LICENSE) file for details.

---
**Author:** [28bc23](https://github.com/28bc23)

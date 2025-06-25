# ViT-D-StyleGAN2-ADA
![Thesis proposal-Integrating ViT Discriminator into StyleGAN2-ADA for Training with Limited Dataset-RAHMAN MD MAHABUBUR-74202309142](https://github.com/user-attachments/assets/f2a02eb4-efb5-49fc-bb38-d9549f2682a6)

A PyTorch-based implementation of the **ViT-StyleGAN2-ADA** framework, designed for image synthesis under limited data scenarios. This project integrates a **multi-scale Vision Transformer (ViT)** discriminator into the **StyleGAN2-ADA** architecture, significantly improving training stability and output quality when only a few thousand training images are available.

## 🧠 Key Features

![image](https://github.com/user-attachments/assets/cc7955af-d880-4afc-9b60-f4cb9fce7c55)

- 🧩 **ViT Discriminator**: Replaces CNN with a Vision Transformer using multi-scale patch processing and grid self-attention.
- 🌀 **Patch Dropout & Shuffle**: Transformer-friendly augmentations integrated into ADA.
- 🔁 **Adaptive Augmentation**: DiffAug + Balanced Consistency Regularization (bCR) dynamically tuned using discriminator feedback.
- 🧪 **Transformer-Specific Losses**: Token-level gradient penalties and class-token Path Length Regularization (PLR).

## 🖼️ Sample Results
Generated samples for the FFHQ (256×256) and AFHQ-CAT, AFHQ-DOG (512×512) datasets, comparing the output of the proposed ViT-D-StyleGAN2-ADA with other methods, including TransGAN and StyleGAN2-ADA. The images demonstrate the quality of the generated samples across different datasets, showcasing the effectiveness of the proposed method:

![image](https://github.com/user-attachments/assets/5192b42c-1c07-420d-a80f-a4d4a4098cc1)


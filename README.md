
# Enhanced LMFA-Net for Real-Time Image Dehazing

This repository contains the implementation of an enhanced version of LMFA-Net for real-time single image dehazing. The project builds upon the original Local Multi-scale Feature Aggregation Network (LMFA-Net) and introduces targeted modifications to improve restoration quality while preserving the lightweight nature of the model. The motivation behind this work is that many efficient dehazing architectures achieve real-time inference but often struggle with structural preservation, noise suppression, and robustness under more challenging degradations. To address these limitations, the proposed framework integrates an adaptive dual-branch restoration mechanism, a channel attention module, and an edge-aware loss function into the baseline architecture.

The task of image dehazing is important in many computer vision applications, including autonomous driving, traffic monitoring, surveillance, robotics, and outdoor scene understanding. Hazy images suffer from reduced contrast, color distortion, and loss of fine structural details, which in turn affect downstream vision tasks such as object detection and semantic segmentation. Although powerful deep networks have demonstrated excellent dehazing performance, many of them are too heavy for real-time use or deployment on resource-constrained devices. This project focuses on improving dehazing quality without sacrificing computational efficiency.

The proposed method is evaluated in three progressively enhanced variants. The first variant is the baseline LMFA-Net, which uses local multi-scale feature aggregation for lightweight haze removal. The second variant extends the baseline by adding an adaptive denoising branch that works in parallel with the dehazing pathway. A learnable gating mechanism dynamically fuses the outputs of these two branches in a spatially adaptive manner. The third and final variant further integrates channel attention using a Squeeze-and-Excitation style module and introduces an edge-aware loss function to improve structural fidelity and perceptual sharpness. These variants are used to perform ablation analysis and study the contribution of each component.

The repository is organized as follows. The `models/` directory contains the network definitions for the baseline and enhanced architectures. The `utils/` directory includes helper utilities used during training, preprocessing, and evaluation. The `PSNR_SSIM/` directory contains scripts or metric implementations used for quantitative assessment. Training is performed using separate scripts such as `train_baseline.py`, `train_adaptive_denoise.py`, and `train_final.py`, while evaluation is performed using `test_baseline.py`, `test_adaptive_denoise.py`, and `test_final.py`. Additional scripts such as `comparative_test_rtts.py` may be used for testing on real-world data, and `video_demo.py` can be used to demonstrate qualitative or real-time behavior on videos. Checkpoint files are expected to be saved inside the `checkpoints/` directory.

Experiments in this project are based on the RESIDE benchmark dataset, which is one of the most widely used datasets for single image dehazing research. The Outdoor Training Set (OTS) is used for supervised training, while the Synthetic Objective Testing Set (SOTS) is used for evaluation. The SOTS Outdoor subset is used as the primary benchmark for reporting quantitative results in this repository. The RESIDE dataset can be downloaded from: https://sites.google.com/view/reside-dehaze-datasets. After downloading, place the training and testing images in the directory structure expected by the training and testing scripts. If needed, update dataset paths inside the scripts before running experiments.

To set up the project, first clone the repository and move into the project folder. Then install the required dependencies using the provided requirements file. A typical setup workflow is:
`git clone https://github.com/AYUSH-1652/Enhanced-LMFA-Net-for-Realtime-Image-Dehazing.git`
`cd Enhanced-LMFA-Net-for-Realtime-Image-Dehazing`
`pip install -r requirements.txt`

If you are using a virtual environment, create and activate it before installing the dependencies. For example, on Windows:
`python -m venv venv`
`venv\Scripts\activate`
`pip install -r requirements.txt`

Training can be performed separately for each model variant depending on the ablation study or experiment you want to reproduce. To train the baseline model, run:
`python train_baseline.py`
To train the adaptive denoising version, run:
`python train_adaptive_denoise.py`
To train the final enhanced model with adaptive denoising, channel attention, and edge-aware loss, run:
`python train_final.py`

During training, checkpoints are typically saved in the `checkpoints/` directory. Ensure that the save path inside the training script matches the folder structure present in your local setup. If the scripts expose configurable parameters such as batch size, learning rate, crop size, dataset location, or number of epochs, adjust them according to your GPU memory and dataset layout. In the experiments reported for this project, the model is trained in PyTorch using the Adam optimizer with a learning rate of `1e-4`, batch size `4`, and training duration of `30` epochs.

Testing is also performed separately for each model variant. To evaluate the baseline model on the SOTS Outdoor dataset, use:
`python test_baseline.py`
To test the adaptive denoising model, use:
`python test_adaptive_denoise.py`
To test the final enhanced model, use:
`python test_final.py`

These scripts load the corresponding checkpoint from the appropriate subdirectory in `checkpoints/` and compute restoration metrics over the test dataset. If the test scripts expect a specific checkpoint path such as `checkpoints/baseline/best.pth`, `checkpoints/adaptive_denoise/best.pth`, or `checkpoints/final/best.pth`, make sure the trained weights are present there. If needed, edit the checkpoint path variable inside the script before running. A typical test run prints progress over the 500 SOTS Outdoor images and finally reports average PSNR and SSIM values.

The project also supports qualitative testing and demo-style execution. If you want to visualize dehazing on real-world or custom images, use the testing scripts after editing input and output paths, or use any dedicated demo script present in the repository. For video-based qualitative evaluation, run:
`python video_demo.py`
If `comparative_test_rtts.py` is included for real-world testing, it can be used to evaluate or compare outputs on non-synthetic hazy images.

The main quantitative results reported on the SOTS Outdoor dataset are as follows. The baseline LMFA-Net achieves a PSNR of `24.75 dB` and an SSIM of `0.9332`. The adaptive denoising LMFA-Net achieves a PSNR of `25.10 dB` and an SSIM of `0.9306`. The final enhanced LMFA-Net achieves the best overall performance with a PSNR of `25.24 dB` and an SSIM of `0.9379`. These results show that while the intermediate denoising variant improves PSNR, the final model provides the best balance of quantitative restoration and structural similarity, which aligns with the goal of improving perceptual quality while preserving lightweight efficiency.

The main contributions of this work are the following. First, an adaptive dual-branch architecture is introduced to jointly perform dehazing and denoising. Second, a lightweight gating mechanism is used to dynamically fuse the outputs of the two restoration branches based on local image content. Third, a channel attention module is incorporated to enhance feature representation and emphasize informative channels. Fourth, an edge-aware loss is introduced to preserve structural details and improve sharpness. Together, these modifications improve restoration quality over the baseline LMFA-Net while keeping the model suitable for real-time applications.

This repository is intended for academic and research purposes. It can serve as a reference implementation for lightweight dehazing research, ablation-based architectural exploration, or future extensions such as video dehazing, domain adaptation, and edge-device deployment. Possible future directions include multi-scale feature modeling for dense haze, real-image generalization through domain adaptation, more efficient attention mechanisms, temporal consistency for video dehazing, and model compression through quantization or pruning.

If you use this repository or build upon this work in your own research, please cite the original LMFA-Net paper as well as this implementation where appropriate. The baseline architecture is based on: Y. Liu and X. Hou, “Local multi-scale feature aggregation network for real-time image dehazing,” Pattern Recognition, vol. 141, p. 109599, 2023.

This project was developed by Rudra Pratap Singh and Ayush Joshi.

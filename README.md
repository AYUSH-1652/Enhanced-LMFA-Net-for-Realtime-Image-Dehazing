# Enhanced LMFA-Net for Real-Time Image Dehazing

This repository contains an enhanced version of LMFA-Net for real-time single image dehazing. The model improves restoration quality while remaining lightweight by introducing an adaptive dual-branch design (dehazing + denoising), channel attention (SE block), and an edge-aware loss for better structural preservation.

## Project Structure
.
├── models/                  
├── utils/                   
├── PSNR_SSIM/               
├── checkpoints/             
├── outputs/                 
├── train_baseline.py        
├── train_adaptive_denoise.py
├── train_final.py           
├── test_baseline.py         
├── test_adaptive_denoise.py 
├── test_final.py            
├── comparative_test_rtts.py 
├── video_demo.py            
├── requirements.txt  
└── README.md  

## Setup
Clone the repository and install dependencies:
git clone https://github.com/AYUSH-1652/Enhanced-LMFA-Net-for-Realtime-Image-Dehazing.git  
cd Enhanced-LMFA-Net-for-Realtime-Image-Dehazing  
pip install -r requirements.txt  

## Dataset
Experiments use the RESIDE dataset:  
- Training: OTS (Outdoor Training Set)  
- Testing: SOTS Outdoor  

Download from:  
https://sites.google.com/view/reside-dehaze-datasets  

Update dataset paths inside scripts if required.

## Training
Train each variant separately:
python train_baseline.py  
python train_adaptive_denoise.py  
python train_final.py  

Checkpoints will be saved in the `checkpoints/` directory.

## Testing
Evaluate models on SOTS dataset:
python test_baseline.py  
python test_adaptive_denoise.py  
python test_final.py  

Each script loads the corresponding checkpoint and reports PSNR and SSIM.

## Additional Usage
Real-world evaluation:
python comparative_test_rtts.py  

Video demo:
python video_demo.py  

## Results (SOTS Outdoor)
Baseline LMFA-Net: PSNR 24.75 | SSIM 0.9332  
Adaptive Denoising LMFA-Net: PSNR 25.10 | SSIM 0.9306  
Final Enhanced LMFA-Net: PSNR 25.24 | SSIM 0.9379  

The final model improves structural similarity while maintaining real-time performance.

## Authors
Rudra Pratap Singh  
Ayush Joshi

## Conclusion

This project enhances LMFA-Net for real-time image dehazing by introducing adaptive dual-branch restoration, channel attention, and an edge-aware loss. These improvements enable better structural preservation and robustness to noise while maintaining a lightweight design. Experimental results on the RESIDE dataset show improved SSIM with competitive PSNR and minimal computational overhead. The model remains suitable for real-time and resource-constrained applications, with future work focusing on real-world generalization, multi-scale modeling, and video dehazing.

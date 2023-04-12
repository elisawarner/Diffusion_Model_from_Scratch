# Diffusion Model for Jupyter Notebook

**Author:** Elisa Warner  
**Email:** elisawa@umich.edu  
**Date:** 04/12/2023  

## Description:
This code is a written implementation of the Diffusion model for Jupyter Notebook.

## Requirements:
1. Python 3.9 or higher  
2. Package: `torchvision`  
3. Package: `torch` (recommended 1.11 or higher)  
4. Package: `os`  
5. Package: `matplotlib`  
6. Package: `jupyter`  
7. For `Preprocess.ipynb`, Package: `glob`, `shutil`  

## Contents:
1. `Preprocess.ipynb` \[Jupyter Notebook\]: This notebook contains code for moving the images downloaded from Kaggle into a single folder.  
2. `DiffusionModel.ipynb` \[Jupyter Notebook\] : This notebook contains the Diffusion Model code.  
3. `unet_mha.py` \[Executable Script\]: This code contains the architecture for the U-Net with Multi-Head Attention. The advantage of this code is that the MHA layers ensure a greater probability that facial landmarks on the cat will be properly placed, but require many more parameters. Therefore, the recommended SQ_SIZE for this network is 32.  
4. `unet_stripped.py` \[Executable Script\]: This code contains the architecture for the U-Net without Multi-Head Attention. The advantage of this code is that the stripped-down model contains less parameters, which means more data can be fit onto the GPU. Therefore, the recommend SQ_SIZE for this network is 64.    
5. `config.py` \[Executable Script\]: This code contains the hyperparameter adjustments set by the user. Edit this code before running `DiffusionModel.ipynb`.  
6. `pre_train_example.pth` : A pretrained 32x32 model example to load. This was trained for over 1200 epochs.  
7. `results_example.txt` : An example output for the model.  

## Expected Outputs:
1. `results.txt` : Will contain the Epoch number as well as the loss.  
2. `model.pth` : The most recently saved model from the latest epoch run on `DiffusionModel.ipynb`.

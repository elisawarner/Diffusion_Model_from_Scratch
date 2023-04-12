import torch
import numpy as np
np.random.seed(100)

#### HYPERPARAMETERS ####
cat_directory = "../Cat-faces-dataset-master/"
EPOCHS = 2000
BETA_START = 1e-4
BETA_END = 1e-2
T = 2000
LR = 1e-2
WD = 0
SQ_SIZE = 32 # how big you want the image to be (e.g. 112x112)
NUM_WORKERS = 2
BATCH_SIZE = 256
RESULTS_OUT = "results.txt"
MODEL_OUT = "model.pth"

###### initialize cuda #######
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION = float(".".join(torch.__version__.split(".")[:2]))

print("Image Size:", SQ_SIZE, "\nBatch Size:", BATCH_SIZE)
print("Device:", device)
print("Pytorch version:", torch.__version__)

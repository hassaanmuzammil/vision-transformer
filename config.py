# config.py

import torch

# dataset params
torchvision_dataset = False
images_folder = './101_ObjectCategories/' # leave empty if torchvision_dataset = True
patch_size = 16 # calculate
num_patches = 256 # calculate
num_classes = 103
train_test_split = 0.9
num_channels = 3 # 3 for rgb else 1 for grayscale

# model params
pre_linear_dim = 512
position_embedding_dim = 512
class_embedding_dim = 512
num_attention_heads = 2

# training params
pre_trained = False
batch_size = 32 # Choose a batch size that is a factor of train and test dataset sizes. 
learning_rate = 1e-4
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

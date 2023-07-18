import torch

# Load the pre-trained model weights
checkpoint_path = 'checkpoint/pretrained/ckpt_semgcn_nonlocal.pth.tar'  # Path to your checkpoint file
weights = torch.load(checkpoint_path)
weights = weights['state_dict']

# Print the keys and shapes of the weights
for key, value in weights.items():
    print(key)
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())

# 安装支持 CUDA 的 PyTorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
import subprocess

import gdown

# Google Drive file ID
file_id = "1Am3LzfitGjDZ3MKCA9yjf1e7-j5_MAd3"
url = f"https://drive.google.com/uc?id={file_id}"
output_path = "C:/Users/hello/OneDrive/Documents/Python/ViT Model/ViT-Model/vit_cifar10.pth"

# Download the model
print("Downloading model from Google Drive...")
gdown.download(url, output_path, quiet=False)
print(f"Model downloaded to: {output_path}")

# Update GitHub repository
repo_dir = "C:/Users/hello/OneDrive/Documents/Python/ViT Model/ViT-Model"
commit_message = "Updated vit_cifar10.pth with latest model"

print("Updating GitHub repository...")
subprocess.run(["git", "-C", repo_dir, "add", "vit_cifar10.pth"])
subprocess.run(["git", "-C", repo_dir, "commit", "-m", commit_message])
subprocess.run(["git", "-C", repo_dir, "push"])

print("Model pushed to GitHub successfully!")

import os
import subprocess  # Allows us to run Git commands from Python

# Path to the GitHub repo (and the location where the model is downloaded)
repo_dir = r"C:/Users/hello/OneDrive/Documents/Python/ViT Model/ViT-Model"

# Step 1: Change the working directory to the repo
os.chdir(repo_dir)

# Step 2: Run Git commands to stage, commit, and push the model
subprocess.run(["git", "add", "vit_cifar10.pth"])  # Stage the updated model
subprocess.run(["git", "commit", "-m", "Updated vit_cifar10.pth with the latest trained model"])  # Commit changes
subprocess.run(["git", "push", "origin", "main"])  # Push to GitHub

#!/usr/bin/env python3

import os
import glob
import shutil
import yaml
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from math import ceil

def setup_logging(log_file):
    logger = logging.getLogger('remove_similar_images')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def get_image_files(input_folder):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    
    if os.path.isdir(input_folder):
        for root, _, files in os.walk(input_folder):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
    elif Path(input_folder).suffix.lower() in image_extensions:
        image_paths.append(input_folder)
        
    return image_paths

def cluster_meanstat(image_paths, threshold):
    from PIL import ImageStat
    
    # Meanstat used difference. In the original script df=0.07 was checked against calculated difference.
    # We will assume threshold is the max allowed difference.
    rms_dict = {}
    
    print("Calculating ImageStat mean for images...")
    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path)
            rms = ImageStat.Stat(img).mean
            rms_dict[img_path] = rms
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    visited = set()
    clusters = []
    
    paths = list(rms_dict.keys())
    for i, path1 in enumerate(tqdm(paths, desc="Clustering")):
        if path1 in visited:
            continue
            
        group = [path1]
        visited.add(path1)
        v1 = rms_dict[path1]
        
        for j, path2 in enumerate(paths[i+1:]):
            if path2 in visited:
                continue
                
            v2 = rms_dict[path2]
            diff = [v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2]]
            
            if (abs(diff[0]) < threshold and 
                abs(diff[1]) < threshold and 
                abs(diff[2]) < threshold):
                group.append(path2)
                visited.add(path2)
                
        if len(group) > 1:
            clusters.append(group)
            
    return clusters

def cluster_phash(image_paths, threshold):
    import imagehash
    
    hash_dict = {}
    print("Calculating perceptual hashes for images...")
    for img_path in tqdm(image_paths):
        try:
            with Image.open(img_path) as img:
                img_hash = imagehash.phash(img)
                hash_dict[img_path] = img_hash
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    visited = set()
    clusters = []
    
    paths = list(hash_dict.keys())
    for i, path1 in enumerate(tqdm(paths, desc="Clustering")):
        if path1 in visited:
            continue
            
        group = [path1]
        visited.add(path1)
        hash1 = hash_dict[path1]
        
        for j, path2 in enumerate(paths[i+1:]):
            if path2 in visited:
                continue
                
            hash2 = hash_dict[path2]
            similarity = (1 - (hash1 - hash2) / len(hash1.hash.flatten())) * 100
            
            if similarity >= threshold:
                group.append(path2)
                visited.add(path2)
                
        if len(group) > 1:
            clusters.append(group)
            
    return clusters

def cluster_resnet(image_paths, threshold, batch_size=32):
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final classification layer
    model.to(device)
    model.eval()
    
    px = 224
    transform = transforms.Compose([
        transforms.Resize((px, px)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    valid_paths = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        path_buffer = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = transform(img)
                images.append(img)
                path_buffer.append(path)
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            batch_features = model(batch_tensor).squeeze(-1).squeeze(-1).cpu().numpy()
            batch_features = batch_features / np.linalg.norm(batch_features, axis=1, keepdims=True)

        features.extend(batch_features)
        valid_paths.extend(path_buffer)
        
    features_np = np.vstack(features)
    visited = np.zeros(len(features_np), dtype=bool)
    clusters = []

    for i in tqdm(range(len(features_np)), desc="Clustering"):
        if visited[i]:
            continue

        # Compute cosine similarity
        sims = np.dot(features_np, features_np[i])
        norms = np.linalg.norm(features_np, axis=1) * np.linalg.norm(features_np[i])
        sims /= norms

        similar_indices = np.where((sims >= threshold) & (~visited))[0]

        if len(similar_indices) > 1:
            cluster = [valid_paths[idx] for idx in similar_indices]
            clusters.append(cluster)
            visited[similar_indices] = True
            
    return clusters

def main():
    cf = Path(__file__).resolve()
    args_file = cf.parents[1] / 'args.yaml'
    
    with open(args_file, 'r') as f:
        conf = yaml.safe_load(f)
        
    config = conf.get('remove_similar_images', {})
    input_folder = config.get('input_folder')
    method = config.get('method', 'resnet')
    similarity_threshold = config.get('similarity_threshold', 0.95)
    
    if not input_folder:
        print("Error: 'input_folder' not found in args.yaml under 'remove_similar_images'")
        return

    print(f"Input Folder: {input_folder}")
    print(f"Method: {method}")
    print(f"Threshold: {similarity_threshold}")
    
    image_paths = get_image_files(input_folder)
    print(f"Found {len(image_paths)} images.")
    
    if not image_paths:
        return

    if method == 'meanstat':
        clusters = cluster_meanstat(image_paths, similarity_threshold)
    elif method == 'phash':
        clusters = cluster_phash(image_paths, similarity_threshold)
    elif method == 'resnet':
        clusters = cluster_resnet(image_paths, similarity_threshold)
    else:
        print(f"Error: Unknown method '{method}'. Valid options are: meanstat, phash, resnet.")
        return
        
    print(f"Found {len(clusters)} clusters containing duplicates.")
    
    duplicate_folder = f"{input_folder}_duplicated"
    os.makedirs(duplicate_folder, exist_ok=True)
    
    log_file = os.path.join(duplicate_folder, "duplicated.log")
    logger = setup_logging(log_file)
    
    moved_count = 0
    for cluster in tqdm(clusters, desc="Moving duplicates"):
        # Keep the first image, move the rest
        for src_path in cluster[1:]:
            try:
                relative_path = os.path.relpath(src_path, input_folder)
                dst_path = os.path.join(duplicate_folder, relative_path)
                
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.move(src_path, dst_path)
                logger.info(f"Moved {src_path} to {dst_path}")
                moved_count += 1
            except Exception as e:
                logger.error(f"Failed to move {src_path}: {e}")
                
    print(f"Done. Moved {moved_count} duplicate images to {duplicate_folder}.")

if __name__ == "__main__":
    main()

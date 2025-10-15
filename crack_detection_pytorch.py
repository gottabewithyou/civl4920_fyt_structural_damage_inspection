import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # List all image and mask files
        image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

        # Helper function to truncate and standardize image bases
        def process_image_name(f):
            base = os.path.splitext(f)[0]
            if base[0].isnumeric():
                return base[:15]
            elif base[0] == "I":
                return base[:8]
            else:
                return base

        # Map processed base name to actual file for images and masks
        image_map = {process_image_name(f): f for f in image_files}
        mask_map = {os.path.splitext(f)[0].removesuffix("_mask"): f for f in mask_files}

        # Find common base names
        common_bases = sorted(set(image_map.keys()) & set(mask_map.keys()))

        # Debug output unmatched files
        unmatched_images = set(image_map.keys()) - set(mask_map.keys())
        unmatched_masks = set(mask_map.keys()) - set(image_map.keys())
        if unmatched_images:
            print(f"Images with no masks: {unmatched_images}")
        if unmatched_masks:
            print(f"Masks with no images: {unmatched_masks}")
        if len(common_bases) == 0:
            raise RuntimeError("No matching image-mask pairs found!")

        # Store actual filenames for loading
        self.image_files = [image_map[b] for b in common_bases]
        self.mask_files = [mask_map[b] for b in common_bases]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Example transforms and usage
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_image_dir = "./datasets/CRACK500/traincrop/traincrop"
train_mask_dir = "./datasets/CRACK500/traindata/traindata"

train_dataset = CrackDataset(train_image_dir, train_mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

for imgs, masks in train_loader:
    print(imgs.shape, masks.shape)
    break

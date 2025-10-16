import os
from PIL import Image
from torch.utils.data import Dataset

class CrackDataset(Dataset):
    """
    PyTorch Dataset for crack detection.
    Handles pairs of images and masks with matching file names.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initialize dataset with image and mask directories and optional transforms.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = self._get_files(image_dir)
        self.mask_files = self._get_files(mask_dir)

        image_bases = set(self._remove_extension(f) for f in self.image_files)
        # Remove the '_mask' suffix from mask base names
        mask_bases = set(self._remove_extension(f).replace('_mask', '') for f in self.mask_files)

        common_bases = sorted(image_bases & mask_bases)

        if not common_bases:
            raise RuntimeError("No matching image-mask pairs found.")

        self.files = []
        for base in common_bases:
            img_file = self._find_file(image_dir, base)
            mask_file = None
            # Mask files have '_mask' suffix in their names
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = os.path.join(mask_dir, base + '_mask' + ext)
                if os.path.isfile(candidate):
                    mask_file = candidate
                    break
            if mask_file is None:
                raise FileNotFoundError(f"No mask file found for base name {base} in {mask_dir}")
            self.files.append((img_file, mask_file))

    def __len__(self):
        """Number of samples"""
        return len(self.files)

    def __getitem__(self, idx):
        """Load and return image-mask pair at given index."""
        img_path, mask_path = self.files[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def _get_files(self, folder):
        """List image files with supported extensions."""
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        return [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]

    def _remove_extension(self, filename):
        """Remove file extension from filename."""
        return os.path.splitext(filename)[0]

    def _find_file(self, folder, base_name):
        """Find file with base_name and any valid image extension in folder."""
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(folder, base_name + ext)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"No file found for base name: {base_name} in {folder}")

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test_crack_dataset():
    # image_dir = "./datasets/CRACK500/traindata/traindata"   # replace with your train image folder
    # mask_dir = "./datasets/CRACK500/traindata/traindata_mask"    # replace with your train mask folder

    image_dir = "./datasets/CRACK500/testdata/testdata"   # replace with your train image folder
    mask_dir = "./datasets/CRACK500/testdata/testdata_mask"    # replace with your train mask folder

    # Define transforms (must convert images and masks to tensors)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = CrackDataset(image_dir, mask_dir, transform=transform)
    print(f"Number of samples: {len(dataset)}")

    # Create dataloader for batch loading
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fetch one batch and display shapes
    images, masks = next(iter(loader))
    print(f"Images shape: {images.shape}")  # Expect [batch_size, 3, 256, 256]
    print(f"Masks shape: {masks.shape}")    # Expect [batch_size, 1, 256, 256]

    # Plot first image and mask pair for visual verification
    img = images[0].permute(1, 2, 0).numpy()
    mask = masks[0].squeeze(0).numpy()

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    test_crack_dataset()

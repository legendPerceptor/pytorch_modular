import random
import os
import shutil
import requests
import zipfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path


NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir:str,
        test_dir:str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int=NUM_WORKERS
):
    """Create training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names),
        where the class_names is a list of the target classes.
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_dataloader, test_dataloader, class_names

def get_subset(image_path = Path("data") / "food-101" / "images",
               data_splits=["train", "test"],
               target_classes=["pizza", "steak", "sushi"],
               amount=0.1,
               seed=42) -> dict:
    """Get a subset of a target dataset.

    Returns:
        A dict that separates "train" and "test" images. The value is a list of image paths.
        Example: {
                "train": ["train/image1.jpg", "train/image2.jpg"]
                "test": ["test/image1.jpg", "test/image2.jpg"]
        }
    """
    random.seed(seed)
    labels_splits = {}
    # Get labels
    for data_split in data_splits:
        image_meta_file = image_path.parent / "meta" / f"{data_split}.txt"
        with open(image_meta_file, 'r') as f:
            all_image_rel_paths = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes]
        
        # Get random subset of target classes image IDs
        number_to_sample = round(amount * len(all_image_rel_paths))
        print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split}...")
        sampled_image_rel_paths = random.sample(all_image_rel_paths, k=number_to_sample)

        image_paths = [Path(str(image_path / sample_image) + ".jpg") for sample_image in sampled_image_rel_paths]
        labels_splits[data_split] = image_paths
    return labels_splits


def copy_subset_images(target_dir: Path, label_splits: dict[str, list[Path]]):
    for image_split in label_splits.keys():
        for image_path in label_splits[str(image_split)]:
            dest_image_path = target_dir / image_split / image_path.parent.stem / image_path.name
            if not dest_image_path.parent.is_dir():
                dest_image_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Copying {image_path} to {dest_image_path}.")
            shutil.copy2(image_path, dest_image_path)
    print("[INFO] Finished copying all images.")


def obtain_food101_dataset(data_dir: str="data"):
    train_data = datasets.Food101(root=data_dir, split="train", download=True)
    test_data = datasets.Food101(root=data_dir, split="test", download=True)
    amount = 0.1
    target_dir = Path(data_dir) / f"pizza_steak_sushi_{amount*100}_percent"
    print(f"Creating directory: {str(target_dir)}")
    target_dir.mkdir(parents=True, exist_ok=True)
    label_splits = get_subset(amount=amount)
    copy_subset_images(target_dir, label_splits)

def directly_get_subset_of_food101_dataset(data_dir: Path):
    image_path = data_dir / "pizza_steak_sushi"
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
    
    # Download pizza, steak, sushi data
    zip_file = data_dir / "pizza_steak_sushi.zip"
    if not zip_file.exists():
        with open(data_dir / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)
    
    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_dir / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)

if __name__ == '__main__':
    obtain_food101_dataset("data")
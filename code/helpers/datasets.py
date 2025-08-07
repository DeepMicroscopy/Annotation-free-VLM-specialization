import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from conch.open_clip_custom import tokenize


class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, fewshot=False, shots=8, seed=None):
        """
        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        if fewshot:
            # Ensure that the dataset is balanced for few-shot learning
            balanced_image_paths = []
            balanced_labels = []
            for label in set(labels):
                indices = [i for i, l in enumerate(labels) if l == label]
                if seed is not None:
                    np.random.seed(seed)
                selected_indices = np.random.choice(indices, shots, replace=False)
                balanced_image_paths.extend([image_paths[i] for i in selected_indices])
                balanced_labels.extend([labels[i] for i in selected_indices])
            self.image_paths = balanced_image_paths
            self.labels = balanced_labels
            print(f"Balanced dataset for few-shot learning: {len(self.image_paths)} samples with {np.unique(self.labels, return_counts=True)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    

def balanced_top_k(df, total):
    """get top k samples with the highest align_score_CONCH from each class
    the class is denoted in the column "class_column_name"
    fill with other classes if minority classes have less than k samples
    balance among classes
    """
    # Group and sort each class by align_score_CONCH descending
    grouped = {
        class_name: group.sort_values("align_score_CONCH", ascending=False)
        for class_name, group in df.groupby("label")
    }

    class_selected_count = {class_name: 0 for class_name in grouped}
    class_remain_count = {class_name: len(group)
                          for class_name, group in grouped.items()}

    total = min(total, len(df))
    selected_total = 0

    # Balanced selection loop
    while selected_total < total:
        for class_name in grouped:
            if class_remain_count[class_name] > 0:
                class_selected_count[class_name] += 1
                class_remain_count[class_name] -= 1
                selected_total += 1
                if selected_total >= total:
                    break
    
    print(f"balanced_top_k: {class_selected_count}")
    # Collect selected rows
    final_df = []
    for class_name, group in grouped.items():
        count = class_selected_count[class_name]
        final_df.append(group.head(count))

    final_df = pd.concat(final_df, ignore_index=True)
    return final_df


class ImageCaptionDataset(Dataset):
    def __init__(self, vl_model, transform, df=None, retrieve_csv=None, tokenizer=None,
                 shots=-1, N=-1, image_dir=None, 
                 return_labels=False, classnames_to_labels={}):
        self.return_labels = return_labels
        if df is None:
            assert retrieve_csv
            df = pd.read_csv(retrieve_csv)

        # select top k most aligned pairs from each class
        if shots > 0:
            num_classes = len(classnames_to_labels)
            df = balanced_top_k(df, total=num_classes*shots)
        # select image-caption pairs that align most
        elif N > 0:
            df = df.sort_values("align_score_CONCH", ascending=False).head(N)
        print(df)
        
        self.images = df["image_path"].tolist()
        self.images = [os.path.join(image_dir, img_path) for img_path in self.images]

        if not return_labels:
            self.captions = df["caption"].tolist()
        else:
            self.labels = df["label"].tolist()
            self.labels = [classnames_to_labels[label] for label in self.labels]

        self.transforms = transform
        self.tokenizer = tokenizer
        self.vl_model = vl_model
        print(f"{len(self.images)} image-caption pairs loaded")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(str(self.images[idx])))
        if self.return_labels:
            return image, self.labels[idx]
        else:
            caption = str(self.captions[idx])
            if self.vl_model == "conch":
                text_tokens = tokenize(texts=[caption], tokenizer=self.tokenizer).squeeze(0)
            elif self.vl_model == "quiltNet":
                text_tokens = self.tokenizer(caption)[0]
            else:
                raise ValueError
            return image, text_tokens


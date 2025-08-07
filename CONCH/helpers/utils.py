import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning.callbacks import Callback
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
from .downstream_dataset_preparation import *

# read_token = "hf_xxx"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    # This enables tf32 on Ampere GPUs which is only 8% slower than
    # float16 and almost as accurate as float32
    # This was a default in pytorch until 1.12
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_model(model_cfg='conch_ViT-B-16'):

    force_image_size = 224
    model, preprocess = create_model_from_pretrained(model_cfg,
                                                     checkpoint_path="hf_hub:MahmoodLab/conch",
                                                    #  hf_auth_token=read_token,
                                                     device=device,
                                                     force_image_size=force_image_size)
    return model, preprocess

def get_conch():
    model, preprocess = get_model()
    tokenizer = get_tokenizer()
    return model, preprocess, tokenizer


def get_quiltnet():
    model_name = 'hf-hub:wisdomik/QuiltNet-B-32'
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess_val, tokenizer


def get_vl_model(vl_model, ckpt=None):
    if vl_model == "conch":
        model, preprocess, tokenizer = get_conch()
    elif vl_model == "quiltNet":
        model, preprocess, tokenizer = get_quiltnet()
    else:
        raise ValueError
    
    if ckpt is not None:
        if not os.path.exists(ckpt):
            raise f"{ckpt} not existing"
        print(f"loading ckpt {ckpt}")
        checkpoint = torch.load(ckpt)
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict.update({k.replace("model.", ""): v})
        model.load_state_dict(state_dict)

    return model, preprocess, tokenizer


class ClipLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features):
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        labels = torch.arange(logits_per_image.shape[0], device=image_features.device, dtype=torch.long)

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss



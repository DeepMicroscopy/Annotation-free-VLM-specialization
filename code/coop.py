# referece: https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py


import argparse
import optuna
from open_clip.transformer import text_global_pool
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from helpers.evaluation import logit_eval
from helpers.datasets import CustomImageDataset
from helpers.utils import *


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.text_pool_type = clip_model.text_pool_type
        self.attn_mask = clip_model.attn_mask.to(device)
        self.dtype = clip_model.transformer.get_cast_dtype()

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.to(self.dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = text_global_pool(x, tokenized_prompts, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, tokenizer, n_ctx, csc=False):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0] # 512

        if not csc:  # unified context with classname at end 
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        else:  # class specific context
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)

        prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors) # to be optimized
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts]) # [n_cls, 77]
        """
        tensor([[49406,   343,   343,   343,   343,   343,   343,   343,   343,   343,
           343,   343,   343,   343,   343,   343,   343,  7997,  6102,  6833,
           335,   269, 49407,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0],
        [49406,   343,   343,   343,   343,   343,   343,   343,   343,   343,
           343,   343,   343,   343,   343,   343,   343,   567,   690,   989,
           803,  8183, 28669,  5358,   269, 49407,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0]])
        """
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # [n_cls, 77, ctx_dim]
        self.register_buffer("token_prefix", embedding[:, :1, :]) # SOS
        self.register_buffer("token_suffix", embedding[:, 1+n_ctx:, :]) # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
    
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # same for each class

        prefix = self.token_prefix  # (n_cls, 1, dim)
        suffix = self.token_suffix # (n_cls, *, dim)
        # class_token_position = "end"
        prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        return prompts
    

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, tokenizer, n_ctx, csc):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, tokenizer, n_ctx, csc)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.transformer.get_cast_dtype()

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))  # [bs, 512]

        prompts = self.prompt_learner() # [n_cls, 77, 512]
        tokenized_prompts = self.tokenized_prompts  # [n_cls, 77]
        text_features = self.text_encoder(prompts, tokenized_prompts)   # [n_cls, 512]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits



class CoOpModelCELoss(pl.LightningModule):
    def __init__(self, model, T_max, lr, wd):
        super().__init__()
        self.model = model
        self.T_max = T_max
        self.lr = lr
        self.wd = wd

    def training_step(self, batch, batch_idx):
        images, labels = batch # (bs, 3, 224, 224), (bs,)
        logits = self.model(images) # [bs, n_cls]
        loss = F.cross_entropy(logits, labels)

        self.log("train_loss", loss, on_epoch=True)

        # lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # self.log("learning_rate", lr, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.prompt_learner.named_parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.T_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def run(args, parameter_tuning=False, lr=0., wd=0.):
    model, preprocess, tokenizer = get_vl_model(args.vl_model)
    model = CustomCLIP(get_coop_classnames(args.task), model, tokenizer, args.n_ctx, args.csc)
    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
        else:
            print(f"Training {name} with shape {param.shape}")
    
    model.to(device)

    filenames, labels = get_task_images_labels(args.task, set="train")
    ds = CustomImageDataset(image_paths=filenames,
                            labels=labels,
                            transform=preprocess,
                            fewshot=True,
                            shots=args.shots,
                            seed=args.seed)
    dl = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    max_epochs = 50
    coop_model = CoOpModelCELoss(model, T_max=max_epochs, lr=lr, wd=wd)

    if not parameter_tuning:
        # lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer(
                            #  callbacks=[lr_monitor],
                             max_epochs=max_epochs,
                             accelerator='gpu',
                             enable_model_summary=1,
                             enable_checkpointing=False,
                             logger=TensorBoardLogger("logs/", name=f"{args.vl_model}", version=f"{args.exp_name}")
                             )
    else:
        trainer = pl.Trainer(max_epochs=5,
                             accelerator='gpu',
                             enable_checkpointing=False,
                             logger=TensorBoardLogger("optuna_logs/", name=f"{args.vl_model}", version=f"{args.exp_name}")
                             )
    trainer.fit(coop_model, train_dataloaders=dl)

    if not parameter_tuning:
        results = logit_eval(args.task, model, preprocess)
        return results
    else:
        return trainer


def parameter_tuning(args):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        trainer = run(args, True, lr, wd)
        return trainer.callback_metrics["train_loss"].item()

    study = optuna.create_study(direction="minimize")  # Minimize loss
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters:", study.best_params)
    return study.best_params


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help='seed for generating few-shot set')
    # Dataset arguments
    parser.add_argument("--task", type=str, default="SICAP")
    parser.add_argument("--image_caption_source", type=str, default="train")
    parser.add_argument("--retrieve_method", type=str, default="train")
    # few-shot arguments
    parser.add_argument('--shots', default=4, type=int)
    # Model arguments
    parser.add_argument("--vl_model", type=str, default="quiltNet", choices=["conch", "quiltNet"])
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    # CoOp arguments
    parser.add_argument('--n_ctx', default=16, type=int, help='number of context words (tokens) to learn')
    parser.add_argument('--csc', default=False, action='store_true', help='use class-specific contexts (CSC) or unified context')

    args = parser.parse_args()

    data_root = 'your_data_root'
    res_folder = "your_res_folder"
    os.makedirs(res_folder, exist_ok=True)

    num_classes = len(get_idx_to_class(args.task))
    args.N = args.shots * num_classes
    args.exp_name = f"coop_{args.vl_model}_{args.task}_n_ctx_{args.n_ctx}_csc_{args.csc}_{args.image_caption_source}_{args.retrieve_method}_shots_{args.shots}_N_{args.N}_seed_{args.seed}"
        
    best_para = parameter_tuning(args)
    results, training_time = run(args, lr=best_para["lr"], wd=best_para["weight_decay"])
    metrics = ["bacc", "acc", "weighted_kappa", "kappa", "roc_auc", "weighted_f1"]
    print([float(np.round(results[m], 4)) for m in metrics])
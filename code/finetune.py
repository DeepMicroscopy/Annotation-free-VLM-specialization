
import argparse
import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from helpers.evaluation import zero_shot_eval
from helpers.datasets import ImageCaptionDataset
from helpers.loralib.utils import *
from helpers.utils import *


class FineTuneModelClipLoss(pl.LightningModule):
    def __init__(self, model, T_max, lr, wd, apply_lora=False):
        super().__init__()
        self.model = model
        self.T_max = T_max
        self.lr = lr
        self.wd = wd
        self.apply_lora = apply_lora
        self.loss_fn = ClipLoss()

    def forward(self, image, text):
        # Encode both images and text
        image_embeddings = self.model.encode_image(image, normalize=True)
        text_embeddings = self.model.encode_text(text, normalize=True)
        return image_embeddings, text_embeddings

    def training_step(self, batch, batch_idx):
        images, captions = batch
        image_embeddings, text_embeddings = self(images, captions)
        loss = self.loss_fn(image_embeddings, text_embeddings)
        self.log("train_loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        if self.apply_lora:
            params = get_lora_parameters(model=self.model)
        else:
            params = self.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.T_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def run(args, parameter_tuning=False, lr=0., wd=0.):
    model, preprocess, tokenizer = get_vl_model(args.vl_model)
    if args.apply_lora:
        list_lora_layers = apply_lora(args, model)
        mark_only_lora_as_trainable(model)

    ds = ImageCaptionDataset(vl_model=args.vl_model,
                             retrieve_csv=args.retrieve_csv,
                             tokenizer=tokenizer,
                             transform=preprocess,
                             shots=args.shots,
                             N=args.N,
                             image_dir=args.image_dir,
                             classnames_to_labels={v: k for k, v in get_idx_to_class(args.task).items()})
    dl = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    max_epochs = 50
    finetune_model = FineTuneModelClipLoss(model, T_max=max_epochs, lr=lr, wd=wd)

    if not parameter_tuning:
        trainer = pl.Trainer(
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
    trainer.fit(finetune_model, train_dataloaders=dl)

    if not parameter_tuning:
        results = zero_shot_eval(vl_model=args.vl_model, task=args.task,
                                 model=model, preprocess=preprocess, tokenizer=tokenizer)
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
    # Dataset arguments
    parser.add_argument("--task", type=str, default="BACH")
    parser.add_argument("--image_caption_source", type=str, default="quilt1m", choices=["ARCH", "quilt1m"])
    parser.add_argument("--retrieve_method", type=str, default="retrieve_classname")
    parser.add_argument('--shots', default=-1, type=int)
    parser.add_argument('--N', default=32, type=int)
    # Model arguments
    parser.add_argument("--vl_model", type=str, default="quiltNet", choices=["conch", "quiltNet"])
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    # LoRA arguments
    parser.add_argument('--apply_lora', default=False, action='store_true', help='LoRA or full finetune')
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA')
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

    args = parser.parse_args()

    data_root = 'your_data_root'
    res_folder = "your_res_folder"
    os.makedirs(res_folder, exist_ok=True)
    retrieve_folder = f"{res_folder}/retrieved/{args.image_caption_source}/{args.task}"
    os.makedirs(retrieve_folder, exist_ok=True)
    os.makedirs(f"{res_folder}/csv_files", exist_ok=True)

    if args.image_caption_source == "ARCH":
        args.image_dir = f"{data_root}/ARCH"
    elif args.image_caption_source == "quilt1m":
        args.image_dir = f"{data_root}/Quilt_1M/quilt_1m_clean"
    else:
        raise ValueError

    num_classes = len(get_idx_to_class(args.task))
    if args.shots > 0:
        args.N = args.shots * num_classes
    args.exp_name = f"loraFT_r{args.r}" if args.apply_lora else "fullFT"
    args.exp_name = f"{args.exp_name}_{args.vl_model}_{args.image_caption_source}_{args.task}_{args.retrieve_method}_shots_{args.shots}_N_{args.N}"
    args.retrieve_csv = f"{retrieve_folder}/{args.retrieve_method}.csv"
    num_retrieved_pairs = len(pd.read_csv(args.retrieve_csv))
    if args.N == -1:
        args.N = num_retrieved_pairs # use all retrieved pairs
    elif args.N > num_retrieved_pairs:
        print(f"skip, args.N ({args.N}) > len(ds) ({num_retrieved_pairs})...")
    else:
        best_para = parameter_tuning(args)
        results = run(args, lr=best_para["lr"], wd=best_para["weight_decay"])
        metrics = ["bacc", "acc", "weighted_kappa", "kappa", "roc_auc", "weighted_f1"]
        print([float(np.round(results[m], 4)) for m in metrics])
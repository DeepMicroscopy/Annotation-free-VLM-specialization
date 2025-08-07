import numpy as np
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             classification_report, roc_auc_score)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from conch.downstream.zeroshot_path import zero_shot_classifier, run_zeroshot
from conch.downstream.utils import AverageMeter

from .datasets import CustomImageDataset
from .downstream_dataset_preparation import *
from .utils import *
from conch.downstream.zeroshot_path import dataloding_post_process

def zero_shot_eval(vl_model, task, model, tokenizer, preprocess):
    filenames, labels = None, None
    
    filenames, labels = get_task_images_labels(task)
    dataset = CustomImageDataset(filenames, labels, preprocess)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=False, num_workers=4)
    print("num samples: ", len(dataloader.dataset))

    classnames, templates = get_classnames_and_template(task)
    _ = model.eval()
    model.to(device)
    zeroshot_weights = zero_shot_classifier(vl_model, model, classnames, templates, tokenizer, device)
    print(zeroshot_weights.shape)

    results, _ = run_zeroshot(model, zeroshot_weights,
                              dataloader, device, dump_results=False)
    print(results)
    return results

@torch.no_grad()
def logit_eval(task, model, preprocess):
    filenames, labels = None, None
    
    filenames, labels = get_task_images_labels(task)
    dataset = CustomImageDataset(filenames, labels, preprocess)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=False, num_workers=4)
    print("num samples: ", len(dataloader.dataset))

    _ = model.eval()
    model.to(device)

    metrics=['acc', 'bacc', 'weighted_kappa', 'kappa', 'roc_auc', 'weighted_f1']
    acc_meter = AverageMeter() 

    logits_all, targets_all, preds_all = [], [], []
    for batch_idx, data in enumerate(tqdm(dataloader)): 
        data = dataloding_post_process(data)
        imgs = data['img'].to(device)
        targets = data['label'].to(device)
        batch_size = targets.size(0)

        logits = model(imgs)
        preds = logits.argmax(dim=1)

        logits_all.append(logits.cpu().numpy())
        targets_all.append(targets.cpu().numpy())
        preds_all.append(preds.cpu().numpy())

        acc_meter.update((preds == targets).float().mean().item(), n=batch_size) # Update AverageMeters with new results

    # Save raw preds & targets
    targets_all = np.concatenate(targets_all, axis=0)
    logits_all = np.concatenate(logits_all, axis=0)
    probs_all = F.softmax(torch.from_numpy(logits_all) * model.logit_scale.exp().item(), dim=1).numpy()
    preds_all = np.concatenate(preds_all, axis=0)
    bacc = balanced_accuracy_score(targets_all, preds_all)
    weighted_kappa = cohen_kappa_score(targets_all, preds_all, weights='quadratic')
    kappa = cohen_kappa_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0) 
    acc = acc_meter.avg

    n_classes = probs_all.shape[1]
    if n_classes == 2:
        class_probs = probs_all[:,1]
        roc_kwargs = {}
    else:
        class_probs = probs_all
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}        
    try:
        roc_auc = roc_auc_score(targets_all, class_probs, **roc_kwargs)
    except ValueError:
        roc_auc = np.nan
    
    print(cls_rep)

    results = {'acc': acc, 
            'bacc': bacc, 
            'weighted_kappa': weighted_kappa,
            'kappa': kappa,
            'roc_auc': roc_auc,
            'weighted_f1': cls_rep['weighted avg']['f1-score']}
    results = {k: results[k] for k in metrics}
    print(results)
    return results
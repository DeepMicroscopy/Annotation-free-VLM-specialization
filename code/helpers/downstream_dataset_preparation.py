
import json
import numpy as np
import os
import pandas as pd

data_root = '/data'

def create_BCAH_dataset():
    idx_to_class = get_idx_to_class("BACH")
    class_to_idx = {v:k for k, v in idx_to_class.items()}

    BCAH_root = f"{data_root}/BACH/Photos"
    ds_csv = BCAH_root.replace("Photos", "microscopy_ground_truth.csv")  # 400 files
    df = pd.read_csv(ds_csv, header=None)
    filenames = df[0].tolist()
    if not filenames[0].__contains__("/"):
        filenames = [BCAH_root+"/"+f for f in filenames]
    labels = df[1].tolist()
    labels = [class_to_idx[cls] for cls in labels]
    print(f"len(filenames) = {len(filenames)}")
    print(np.unique(labels, return_counts=True))
    return filenames, labels


def prepare_SICAP_dataset():
    SICAP_root = f"{data_root}/SICAPv2"

    df_final = pd.DataFrame(columns=["image_name", "label", "IsTest"])
    label_csv = f"{SICAP_root}/image_label.csv"
    for set in ["Train", "Test"]:
        df = pd.read_excel(os.path.join(
            SICAP_root, "partition", "Test", f"{set}.xlsx"))   # 9959 train, 2122 test
        df["label"] = -1
        df["IsTest"] = 1 if set == "Test" else 0
        for idx, cls in enumerate(["NC", "G3", "G4", "G5"]):
            df.loc[df[cls] == 1, "label"] = idx
        df.drop(columns=["NC", "G3", "G4", "G5", "G4C"], inplace=True)
        print(f"len(df) = {len(df)}")
        df_final = pd.concat([df_final, df])
        print(f"len(df_final) = {len(df_final)}")
        print(np.unique(df["label"].to_numpy(), return_counts=True))
    df_final.to_csv(label_csv, index=False)

def create_SICAP_dataset(set="test"):
    assert set in ["train", "test"]
    SICAP_root = f"{data_root}/SICAPv2"
    ds_csv = os.path.join(SICAP_root, "image_label.csv")
    df = pd.read_csv(ds_csv)
    if set == "test":
        df = df[df["IsTest"] == 1]
    else:
        df = df[df["IsTest"] == 0]
    filenames = df["image_name"].tolist()
    if not filenames[0].__contains__("/"):
        filenames = [SICAP_root+"/images/"+f for f in filenames]
    # check_existence = [f for f in filenames if not os.path.exists(f)]
    # print(f"the following files do not exist: {check_existence}")
    labels = df["label"].tolist()
    print(f"len(filenames) = {len(filenames)}")
    print(np.unique(labels, return_counts=True))
    return filenames, labels

def create_MHIST_dataset(set="test"):
    assert set in ["train", "test"]
    MHIST_root = f"{data_root}/MHIST"
    ds_csv = os.path.join(MHIST_root, "annotations.csv")
    df = pd.read_csv(ds_csv)
    df = df[df["Partition"] == set]
    filenames = df["Image Name"].tolist()
    if not filenames[0].__contains__("/"):
        filenames = [MHIST_root+"/images/"+f for f in filenames]
    # check_existence = [f for f in filenames if not os.path.exists(f)]
    # print(f"the following files do not exist: {check_existence}")
    labels = df["Majority Vote Label"].tolist()
    labels = [1 if ele == "SSA" else 0 for ele in labels]
    print(f"len(filenames) = {len(filenames)}")
    print(np.unique(labels, return_counts=True))
    return filenames, labels


def get_task_images_labels(task, set="test"):
    if task == "BACH":
        filenames, labels = create_BCAH_dataset()
    elif task == "SICAP":
        filenames, labels = create_SICAP_dataset(set)
    elif task == "MHIST":
        filenames, labels = create_MHIST_dataset(set)
    else:
        raise ValueError
    
    return filenames, labels


def get_idx_to_class(task):
    
    if task == "BACH":
        idx_to_class = {0: "Normal", 1: "Benign", 2: "InSitu", 3: "Invasive"}
    elif task == "MHIST":
        idx_to_class = {0: "HP", 1: "SSA"}
    elif task == "SICAP":
        idx_to_class = {0: "NC", 1: "G3", 2: "G4", 3: "G5"}
    else:
        raise ValueError
    return idx_to_class


def get_task_keywords(task):
    if task == "BACH":
        description = "breast cancer classification"
        organ = ["breast"]
        class_keys = [" normal ", "benign", "in situ",
                      "insitu", "in-situ", "invasive"]
    
    elif task == "MHIST":
        description = "colorectal polyp classification"
        organ = ["colon", "colorectal", "polyp"]
        class_keys = ["hyperplastic", "benign", "sessile", "serrated", "adenoma"]

    elif task == "SICAP":
        description = "prostate cancer diagnosis based on the Gleason grading system"
        organ = ["prostate", "gland"]
        class_keys = ["non-cancerous", "gleason"]

    return description, organ, class_keys


def get_prompts(task):
    prompt_file = f'CONCH/prompts/{task}_prompts_all_per_class.json'
    with open(prompt_file) as f:
        prompts = json.load(f)['0']
    return prompts


def get_classnames_and_template(task):
    idx_to_class = get_idx_to_class(task)
    prompts = get_prompts(task)
    classnames = prompts['classnames']
    templates = prompts['templates']
    n_classes = len(classnames)
    classnames = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
    for class_idx, classname in enumerate(classnames):
        print(f'{class_idx}: {classname}')

    return classnames, templates


def get_coop_classnames(task):
    if task == "BACH":
        classnames = ["normal tissue", 
                      "benign tissue", 
                      "in situ carcinoma", 
                      "invasive carcinoma"]
    elif task == "MHIST":
        classnames = ["hyperplastic polyp",
                      "sessile serrated adenoma"]
    elif task == "SICAP":
        classnames = ["non-cancerous prostate tissue",
                      "Gleason 3 prostate cancer",
                      "Gleason 4 prostate cancer",
                      "Gleason 5 prostate cancer"]
    
    return classnames
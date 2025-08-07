from glob import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoModel
from conch.open_clip_custom import get_tokenizer, tokenize

from .downstream_dataset_preparation import *
from .utils import *
from .datasets import ImageCaptionDataset


def get_caption_df(filename):
    if filename.endswith("json"):
        with open(filename, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient='index')[['uuid', 'caption']]
        df.rename(columns={"uuid": "image_path"}, inplace=True)
    elif filename.endswith("csv"):
        df = pd.read_csv(filename)
    else:
        raise ValueError

    return df


def load_Gemma():
    ckpt = "google/gemma-3-4b-it"
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    # For disabling warnings.
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    processor = AutoProcessor.from_pretrained(ckpt)
    return model, processor

@torch.no_grad()
def apply_Gemma(model, processor, messages, max_new_tokens=100):
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generation = generation[0, input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded.replace('\n', '').strip()


def caption_classify_with_Gemma(df, task_description, classnames_synonyms):
    """classnames_synonyms:
    {
            "Normal": [
                "non-tumor",
                "normal tissue",
                "non-cancerous tissue"
            ],
            "Benign": [
                "non-malignant tissue",
                "benign tissue",
                "non-malignant benign tissue"
            ],
            "InSitu": [
                "in-situ tumor",
                "in-situ cancer",
                "in-situ carcinoma",
                "malignant in-situ carcinoma"
            ],
            "Invasive": [
                "invasive tumor",
                "invasive cancer",
                "invasive carcinoma",
                "malignant invasive carcinoma"
            ]
        }
    """
    description_lines = "\n".join(
        [f"- {label}: {', '.join(descriptions)}" for label,
         descriptions in classnames_synonyms.items()])
    print(description_lines)
    model, processor = load_Gemma()
    res = []

    # a walkaround of the problem that Gemma cannot process too many requests, store temporary results
    if os.path.exists("results/tmp.csv"):
        tmp_df = pd.read_csv("results/tmp.csv")
        start = len(tmp_df)
    else:
        start = 0
        with open("results/tmp.csv", "w") as f:
            f.write("idx,label\n")
        
    for idx, caption in enumerate(df["caption"]):
        if idx < start:
            continue    
        if idx % 100 == 0:
            print(f"{idx}/{len(df)}")

        prompt = f"""
            You are given a text associated with histopathology images.
            Determine whether the text is relevant to a {task_description} task that involves the following {len(classnames_synonyms)} categories, each with associated indicative phrases:

            Categories: {description_lines}

            Input text: "{caption}"

            If the text is not relevant to the {task_description} task, respond with "not-relevant". 
            
            Respond with one of: {list(classnames_synonyms.keys())+["not-relevant"]}.
            """

        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful text classifier."},]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt.strip()}]
                },
            ],
        ]

        res.append(apply_Gemma(model, processor, messages, max_new_tokens=5))
        with open("results/tmp.csv", "a") as f:
            f.write(f"{idx},{res[-1]}\n")
    
    tmp_df = pd.read_csv("results/tmp.csv")
    res = tmp_df["label"].to_list()

    df["label"] = res
    df = df[df['label'].isin(list(classnames_synonyms.keys()))]
    print(np.unique(df["label"], return_counts=True))
    return df


def caption_split_with_Gemma(model, processor, caption, identifier):
    prompt = f"""
        Your task is to extract the specific sub-caption associated with a given identifier (e.g., A, B, etc.) from a combined caption string. Sub-captions are typically separated by these identifiers, which may appear as plain letters (A, B, ...) or in parentheses (e.g., (a)). Identifiers can be placed at the start or end of a sub-caption.

        If a sub-caption corresponds to multiple identifiers, assign it to each relevant one. Include any general information that applies to all sub-captions, but exclude content clearly linked to other identifiers.

        Caption: {caption}
        Target Identifier: {identifier}

        Return only the extracted sub-caption text corresponding to the specified identifier.
        """
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful text spliter."},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt.strip()}]
            },
        ],
    ]
    return apply_Gemma(model, processor, messages, max_new_tokens=100)


def ARCH_bookset_caption_split(caption_json, res_json):
    model, processor = load_Gemma()
    res = {}
    with open(caption_json) as f:
        caption_dict = json.load(f)
    for key, value in caption_dict.items():
        caption = value["caption"]
        if not value["letter"] == "Single":
            caption = caption_split_with_Gemma(model, processor, caption, identifier=value["letter"])
        res.update({key: {"caption": caption, "uuid": value["uuid"]}})
        
    with open(res_json, "w") as f:
        json.dump(res, f, indent=2)



def quilt_1M_data_clean(quilt_root):
    df = pd.read_csv(f"{quilt_root}/quilt_1M_lookup.csv")
    """
    ['Unnamed: 0', 'caption', 'image_path', 'subset', 'split', 'pathology',
        'roi_text', 'noisy_text', 'corrected_text', 'med_umls_ids',
        'magnification', 'height', 'width']
    """
    # step 1: make the lookup table smaller by keeping only image_path and caption
    columns_to_keep = ["image_path", "caption"]
    new_df = df[columns_to_keep]
    # length: 1017712
    new_df.to_csv(f"{quilt_root}/quilt_1M_lookup_filename_caption_only.csv")

    # step 2: make the lookup table smaller by keeping only clean images
    df_cleaner = pd.read_csv(f"{quilt_root}/predictions_quiltcleaner.csv")
    """
    ['Unnamed: 0', 'Persons/Photos', 'Desktop/Windows/SlideViewer', 'Text/Logo in Image', 'Arrow/Annotation', 'Image Perspective/Quality', 'Additional (On-Slide) Overview', 'Additional Control Elements', 'Multi-Panel Image',
        'Any impairment', 'Filename']
    """
    df_cleaner_train = pd.read_csv(f"{quilt_root}/train_annotations.csv")
    df_cleaner_val = pd.read_csv(f"{quilt_root}/val_annotations.csv")
    df_cleaner_test = pd.read_csv(f"{quilt_root}/test_annotations.csv")
    """
    ['Image', 'Persons/Photos', 'Desktop/Windows/SlideViewer', 'Text/Logo in Image', 'Arrow/Annotation', 'Image Perspective/Quality', 'Additional (On-Slide) Overview', 'Additional Control Elements', 'Multi-Panel Image']
    """

    clean_images = []
    for index, row in df_cleaner.iterrows():
        if row["Any impairment"] < 0.5:
            clean_images.append(row["Filename"])

    for df in [df_cleaner_train, df_cleaner_val, df_cleaner_test]:
        for index, row in df.iterrows():
            if not any(row[['Persons/Photos', 'Desktop/Windows/SlideViewer', 'Text/Logo in Image', 'Arrow/Annotation', 'Image Perspective/Quality', 'Additional (On-Slide) Overview', 'Additional Control Elements', 'Multi-Panel Image']].to_numpy()):
                clean_images.append(row["Image"].replace("images/", ""))
    print(f"len(clean_images): {len(clean_images)}")
    with open(f"{quilt_root}/clean_images.json", 'w') as f:
        json.dump(clean_images, f)

    csv_file = f"{quilt_root}/quilt_1M_lookup_filename_caption_only.csv"
    df = pd.read_csv(csv_file)
    filtered_rows = df[df["image_path"].isin(clean_images)]
    print(len(filtered_rows))
    pd.DataFrame(filtered_rows).to_csv(
        f"{quilt_root}/quilt_1M_lookup_filename_caption_clean.csv", index=False)  # lengeh: 232039


def load_PathologyBERT():
    model_name = "tsantos/PathologyBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def text_embedding_PathologyBERT(model, tokenizer, texts):
    inputs = tokenizer(texts,
                       return_tensors="pt",
                       truncation=True,
                       max_length=512,
                       padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # [num_text, hidden_size]
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    cls_embedding = normalize(cls_embedding, p=2, dim=1)

    return cls_embedding


@torch.no_grad()
def calculate_prototype_embeddings(model_name, task, prototype_hdf5_path):
    class_prototypes = []
    
    if os.path.exists(prototype_hdf5_path):
        with h5py.File(prototype_hdf5_path, "r") as f:
            class_prototypes = f["embeddings"][:]

    if len(class_prototypes) > 0:
        print(f"{prototype_hdf5_path} has been calculated.")
    else:
        classnames_synonyms, templates = get_classnames_and_template(task)
        
        if model_name == "PathologyBERT":
            model, tokenizer = load_PathologyBERT()
        elif model_name == "CONCHTextEncoder":
            model, _ = get_model()
            model.eval()
            model.to(device)
            tokenizer = get_tokenizer()
        else:
            raise NotImplementedError
        
        # e.g, ["non-tumor", "normal tissue", "non-cancerous tissue"]
        for synonyms in classnames_synonyms:
            class_embedding = []
            for s in synonyms:
                texts = [template.replace('CLASSNAME', s) for template in templates]

                if model_name == "PathologyBERT":
                    texts_embeddings = text_embedding_PathologyBERT(model, tokenizer, texts)  # [num_templates, embedding_dim]
                elif model_name == "CONCHTextEncoder":
                    token_ids = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
                    texts_embeddings = model.encode_text(token_ids.to(device), normalize=True)
                else:
                    raise NotImplementedError
                
                class_embedding.append(texts_embeddings)

            # [num_classnames, num_templates, embedding_dim]
            class_embedding = torch.stack(class_embedding, dim=0)
            class_embedding = class_embedding.mean(dim=(0, 1))  # [embedding_dim]
            class_embedding /= class_embedding.norm()
            class_prototypes.append(class_embedding)

        # [num_classes, embedding_dim]
        class_prototypes = torch.stack(class_prototypes, dim=0).cpu().numpy()  # [num_classes, embedding_dim]

        store = h5py.File(prototype_hdf5_path, "a")
        store.create_dataset("embeddings", data=class_prototypes, compression="gzip")
        store.close()


@torch.no_grad()
def calculate_caption_embeddings(model_name, df, caption_hdf5_path):
    embedding_pool = []

    if os.path.exists(caption_hdf5_path):
        with h5py.File(caption_hdf5_path, "r") as f:
            embedding_pool = f["embeddings"][:]

    if len(embedding_pool) > 0:
        print(f"{caption_hdf5_path} has been calculated.")
    else:
        batch_size = 64
        if model_name == "PathologyBERT":
            model, tokenizer = load_PathologyBERT()
            hidden_size = model.config.hidden_size
        elif model_name == "CONCHTextEncoder":
            model, _ = get_model()
            tokenizer = get_tokenizer()
            model.eval()
            model.to(device)
            hidden_size = model.text.output_dim
        else:
            raise NotImplementedError
        

        with h5py.File(caption_hdf5_path, "w") as f:
            embedding_pool = f.create_dataset("embeddings", shape=(len(df), hidden_size), dtype=np.float32)

            for i in np.arange(0, len(df), batch_size):
                texts = df["caption"][i:i+batch_size].to_list()
                if model_name == "PathologyBERT":
                    texts_embeddings = text_embedding_PathologyBERT(model, tokenizer, texts)  # [num_templates, embedding_dim]
                elif model_name == "CONCHTextEncoder":
                    token_ids = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
                    texts_embeddings = model.encode_text(token_ids.to(device), normalize=True)
                else:
                    raise NotImplementedError
                
                embedding_pool[i:i+len(texts)] = texts_embeddings.cpu().numpy()


def caption_classify_with_embeddings(df, idx_to_class, prototype_hdf5_path, caption_hdf5_path):
    with h5py.File(prototype_hdf5_path, "r") as f:
        class_prototypes = f["embeddings"][:]  # (num_classes, embed_dim)

    with h5py.File(caption_hdf5_path, "r") as f:
        embedding_pool = f["embeddings"][:]  # (num_captions, embed_dim)

    similarities = embedding_pool @ class_prototypes.T  # (num_captions, num_classes)
    labels = np.argmax(similarities, axis=1)
    labels = [idx_to_class[l] for l in labels]
    print(np.unique(labels, return_counts=True))
    df["label"] = labels
    return df

def get_suffix(image_dir, df):
    for idx, row in df.iterrows():
        img_path_existed = glob(str(os.path.join(image_dir, row["image_path"])) + ".*")
        if len(img_path_existed) > 0:
            suffix = img_path_existed[0].split(".")[1]
            df._set_value(idx, "image_path", row["image_path"] + f".{suffix}")
        else:
            df = df.drop(idx)
            print(f"{row['image_path']} not found")
    return df

def check_image_exists(image_dir, df):
    for idx, row in df.iterrows():
        if not os.path.exists(os.path.join(image_dir, row["image_path"])):
            df = df.drop(idx)
            print(f"{row['image_path']} not found")
    return df


@torch.no_grad()
def semantic_sort_pair_with_CONCH(df, image_dir):
    model, preprocess = get_model()
    tokenizer = get_tokenizer()
    ds = ImageCaptionDataset(vl_model="conch",
                             df=df,
                             tokenizer=tokenizer,
                             transform=preprocess,
                             image_dir=image_dir)
    dl = DataLoader(ds, batch_size=8, shuffle=False,
                    num_workers=4, pin_memory=True, drop_last=False)

    _ = model.eval()
    scores = []
    for batch_idx, batch in enumerate(dl):
        images, captions = batch
        image_embeddings = model.encode_image(images.to(device))
        text_embeddings = model.encode_text(captions.to(device))
        scores.append(torch.diag(image_embeddings @ text_embeddings.T).clone().detach())

    scores = torch.concat(scores).cpu().numpy()
    df["align_score_CONCH"] = scores
    df = df.sort_values('align_score_CONCH', ascending=False)
    return df


    # fig, axes = plt.subplots(2, 4)
    # axes = axes.ravel()
    # for idx, ax in enumerate(axes):
    #     img_path = os.path.join(image_dir, df_sorted[idx:idx+1]["image_path"].item())
    #     img = Image.open(str(img_path))
    #     caption = df_sorted[idx:idx+1]["caption"].item()
    #     ax.imshow(img)
    #     ax.set_title(caption)
    # fig.savefig("results/tmp.png")
    # plt.close()


def plot_retrieved_images(retrieve_csv):
    df = pd.read_csv(retrieve_csv)[:25]
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.ravel()
    for ax in axes:
        ax.set_axis_off()
    for idx, image_path in enumerate(df["image_path"]):
        axes[idx].imshow(plt.imread(image_path))
    plt.tight_layout()
    plt.savefig(retrieve_csv.replace(".csv", "_visualize.png"))
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reserve_top_similar(res_csv, K):
    df = pd.read_csv(res_csv)
    sorted_df = df.sort_values(
        by=["class_idx", "similarity"], ascending=[True, False])
    reserve = sorted_df.groupby("class_idx").head(K).reset_index(drop=True)
    reserve["K"] = list(np.arange(1, K+1)) * np.max(df["class_idx"])
    reserve.to_csv(res_csv, index=False)


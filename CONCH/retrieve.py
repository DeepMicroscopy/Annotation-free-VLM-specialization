from helpers.retrieve_utils import *


def write_to_csv(df, retrieve_csv):
    order = ['image_path', 'label', "similarity", 'align_score_CONCH', 'caption']
    order = [col for col in order if col in df.columns]
    df = df.reindex(columns=order)
    df.to_csv(retrieve_csv, index=False)


def retrieve_image_caption_pairs(task, image_caption_source,
                                 caption_dir, image_dir, save_path,
                                 string_match=True,
                                 Gemma_classify=False, pre_retrieve_csv=None,
                                 PathologyBERT_classify=False, 
                                 prototype_hdf5_path=None, caption_hdf5_path=None):
    description, organ, class_keys = get_task_keywords(task)
    if string_match:    
        retrieve_csv = str(os.path.join(save_path, "retrieve_organ.csv"))
        df = get_caption_df(caption_dir)
        df = df[df['caption'].str.contains('|'.join(organ), case=False)]
        print(f"organ_match: {len(df)}")

        if image_caption_source == "ARCH":
            df = get_suffix(image_dir, df)
        else:
            df = check_image_exists(image_dir, df)
        df = semantic_sort_pair_with_CONCH(df, image_dir=image_dir)
        write_to_csv(df, retrieve_csv)

        df = df[df['caption'].str.contains('|'.join(class_keys), case=False)]
        print(f"classname_match: {len(df)}")
        retrieve_csv = str(os.path.join(save_path, "retrieve_classname.csv"))
        write_to_csv(df, retrieve_csv)

    else:
        assert pre_retrieve_csv is not None
        df = pd.read_csv(pre_retrieve_csv)

    if Gemma_classify:
        retrieve_csv = pre_retrieve_csv.replace(".csv", "_Gemma.csv")
        df = caption_classify_with_Gemma(df, description, get_prompts(task)['classnames'])
        print(f"Gemma_classify: {len(df)}")
        write_to_csv(df, retrieve_csv)
        
    if PathologyBERT_classify:
        model_name = "PathologyBERT"
        retrieve_csv = pre_retrieve_csv.replace(".csv", f"_{model_name}.csv")
        calculate_prototype_embeddings(model_name, task, prototype_hdf5_path)
        calculate_caption_embeddings(model_name, df, caption_hdf5_path)
        caption_classify_with_embeddings(df, get_idx_to_class(task), prototype_hdf5_path, caption_hdf5_path)
        write_to_csv(df, retrieve_csv)

if __name__ == "__main__":

    for image_caption_source in [
        "ARCH",
        "quilt1m"
    ]:
        if image_caption_source == "ARCH":
            root = "data/ARCH"
            image_dir = root
            caption_dir = f"{root}/captions.json"
        elif image_caption_source == "quilt1m":
            root = "data/Quilt_1M"
            image_dir = f"{root}/quilt_1m_clean"
            caption_dir = f"{root}/quilt_1M_lookup_filename_caption_clean.csv"
        else:
            raise ValueError
        
        for task in [
            "BACH",
            "MHIST",
            "SICAP"
        ]:
            save_path = f"results/retrieved/{image_caption_source}/{task}"
            os.makedirs(save_path, exist_ok=True)
            retrieve_method = "retrieve_classname"
            pre_retrieve_csv = str(os.path.join(save_path, f"{retrieve_method}.csv"))

            PathologyBERT_classify = False
            model_name = "PathologyBERT"

            prototype_hdf5_path = f"results/prototype_{model_name}_{task}.h5"
            caption_hdf5_path = f"results/{image_caption_source}_{task}_{retrieve_method}_{model_name}.h5"
            retrieve_image_caption_pairs(task, image_caption_source, 
                                         caption_dir, image_dir, save_path,
                                         string_match=True,
                                         Gemma_classify=False, 
                                         pre_retrieve_csv=pre_retrieve_csv,
                                         PathologyBERT_classify=PathologyBERT_classify,
                                         prototype_hdf5_path=prototype_hdf5_path, 
                                         caption_hdf5_path=caption_hdf5_path)

import json
import numpy as np
import os
import torch
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import requests
from torch.utils.data import Dataset, DataLoader, get_worker_info
from transformers.image_utils import to_numpy_array


SEED = 42
IMG_TOKEN = '<image>'
DATAROOT = Path('/import/ml-sc-scratch5/vamsik/mm_ml_llms/datasets/visual_genome_jap')
DATAROOT.mkdir(exist_ok=True)
IMGROOT = Path('/import/ml-sc-scratch1/jonathanl/data/visual_genome/') # For VG dataset, the images cannot be downloaded from HF, they are seperately downloaded from https://huggingface.co/datasets/llm-jp/ja-vg-vqa-conversation 

def validate_image(img_path):
    if img_path is not None and os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            img_array = to_numpy_array(img)
            if img.size[0] >= 50  and img.size[1] >= 50:
                return True
            else:
                os.remove(img_path)
        except:
            os.remove(img_path)

    return False

# Load hf dataset
if __name__ == '__main__':
    DATAROOT.mkdir(exist_ok=True)
    metadata_dict = dict()
    split = 'train'
    ds = load_dataset('llm-jp/ja-vg-vqa-conversation', split=split, cache_dir='/import/ml-sc-scratch5/vamsik/hf_cache')
    dataset_len = len(ds) 
    # iterate through ds.
    # save it in json form.
    json_entries = []
    for idx in range(dataset_len):
        datapoint = ds[idx]
        data_id = datapoint['id']
        img_local_path = '/'.join(datapoint['image'].split('/')[1:])
        img_full_path = IMGROOT / img_local_path
        conversations = datapoint['conversations'] # make sure the conversations are according to the template.

        is_img_valid = validate_image(img_full_path)

        if is_img_valid:
            for conv_id in range(len(conversations)):
                conv = conversations[conv_id]
                conv["from"] = "user" if conv["from"] == "human" else "assistant"
                conv['role'] = conv.pop('from')
                conv['content'] = conv.pop('value')

        json_entries.append({'messages': conversations, 'images': [str(img_full_path)]})

    # dump the json file.
    with open(DATAROOT / 'visual_genome_jap_01_24_2025.json', 'w') as json_file:
        json.dump(json_entries, json_file, indent=2)

    print(f'Done writing Visual Genome dataset json!', flush=True)
        

# make a json file if the images are loadable

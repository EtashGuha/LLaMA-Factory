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
DATAROOT = Path('/import/ml-sc-scratch5/vamsik/mm_ml_llms/datasets/PALO')
DATAROOT.mkdir(exist_ok=True)
IMGROOT = DATAROOT / 'data'
IMGROOT.mkdir(exist_ok=True)

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

    ds = []
    with open('/import/ml-sc-scratch2/jonathanl/data/palo_multilingual_dataset/palo_multilingual_dataset.json', 'r') as json_file:
        ds = json.load(json_file)

    dataset_len = len(ds) 
    # iterate through ds.
    # save it in json form.
    json_entries = []
    for idx in range(dataset_len):
        datapoint = ds[idx]
        data_id = datapoint['id']
        print(idx, datapoint.keys())

        is_japanese_datapoint = 'japanese_conversation' in datapoint.keys()
        has_image_path = 'image' in datapoint.keys()
        
        if not (is_japanese_datapoint and has_image_path):
            continue

        img_local_path = datapoint['image']
        is_vg_datapoint = 'vg/VG_100K' in datapoint['image'] # Excluding Visual Genomics (VG) datapoints from the PALO dataset, as this dataset is already included elsewhere.

        if is_vg_datapoint:
            continue

        conversations = datapoint['japanese_conversation'] # make sure the conversations are according to the template.

        img_full_path = IMGROOT / img_local_path

        is_img_valid = validate_image(img_full_path)

        if is_img_valid:
            for conv_id in range(len(conversations)):
                conv = conversations[conv_id]
                conv['role'] = conv.pop('from')
                conv['content'] = conv.pop('value')
                conv["role"] = "user" if conv["role"] == "human" else "assistant"

                content_list = conv['content'].split('\n<image>')
                is_img_token_in_back = content_list[-1] == '\n<image>'
                if is_img_token_in_back:
                    content = '<image>\n' + content_list[0]
                    conv['content'] = content

        json_entries.append({'messages': conversations, 'images': [str(img_full_path)]})

    # dump the json file.
    with open(DATAROOT / 'palo_non_vg_dataset_jap_02_03_2025.json', 'w') as json_file:
        json.dump(json_entries, json_file, indent=2)

    print(f'Done writing PALO dataset json!', flush=True)
        

# make a json file if the images are loadable

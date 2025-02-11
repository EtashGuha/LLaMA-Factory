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


IMG_TOKEN = '<image>'
DATAROOT = Path('/import/ml-sc-scratch5/vamsik/mm_ml_llms/datasets/docqa_nonbinary_jap')
DATAROOT.mkdir(exist_ok=True)

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
    splits = ['train', 'test', 'val']

    for split in splits:
        split_dir = DATAROOT / split
        image_dir =  DATAROOT / split / 'images'
        image_dir.mkdir(exist_ok=True, parents=True)
        ds = load_dataset('jlli/JDocQA-nonbinary', split=split, cache_dir='/import/ml-sc-scratch5/vamsik/hf_cache')
        dataset_len = len(ds) 
        # iterate through ds.
        # save it in json form.
        json_entries = []
        for idx in range(dataset_len):
            datapoint = ds[idx]
            image = datapoint['image']
            question = datapoint['question']
            answer = datapoint['original_answer']
            text = datapoint['text'] # OCR reading of the image.
            question_text = IMG_TOKEN + '\n' + question

            img_full_path = image_dir / f'{idx}.jpg'

            # download image.
            try:
                image.save(img_full_path)
            except:
                continue

            is_img_valid = validate_image(img_full_path)

            conversations = []
            if is_img_valid:
                conversations.append({"content": question_text, "role": "user"})
                conversations.append({"content": answer, "role": "assistant"})

            json_entries.append({'messages': conversations, 'images': [str(img_full_path)]})

        # dump the json file.
        with open(split_dir / f'docqa_nonbinary_jap_{split}_01_27_2025.json', 'w') as json_file:
            json.dump(json_entries, json_file, indent=2)

        print(f'Done writing Japanese non binary DocQA {split} dataset json!', flush=True)
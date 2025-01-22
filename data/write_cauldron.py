from datasets import load_dataset
from pathlib import Path
import io
import os
import json
import jsonlines
import math
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

DATAROOT = Path('/import/ml-sc-scratch1/jonathanl/repositories/EtashGuha-LLaMA-Factory/data')
HF_DS_NAME = 'HuggingFaceM4/the_cauldron'
SEED = 42
IMG_TOKEN = '<image>'

def get_splits(ds_name):
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is None:
        raise ValueError(
            "HF token not found. Please set the HF_TOKEN environment variable."
        )
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.get(f'https://datasets-server.huggingface.co/splits?dataset={ds_name}', headers=headers)
    data = response.json()
    configs_and_splits = [(splitdict['config'], splitdict['split']) for splitdict in data['splits']]
    return configs_and_splits

if __name__ == '__main__':
    DATAROOT.mkdir(exist_ok=True)
    metadata_dict = dict()
    conversations_per_split = defaultdict(list)
    configs_and_splits = get_splits(HF_DS_NAME)
    state = np.random.RandomState(seed=SEED)
    for ds_idx, (ds_config, split) in enumerate(configs_and_splits):
        ds = load_dataset(HF_DS_NAME, ds_config, split=split, cache_dir='/import/ml-sc-scratch3/jonathanl/cache')
        img_root = DATAROOT / 'the_cauldron' / ds_config / split
        img_root.mkdir(parents=True, exist_ok=True)
        for i, datapoint in tqdm(enumerate(ds), total=len(ds), dynamic_ncols=True, desc=f'{ds_config} {split} split ({ds_idx}/{len(configs_and_splits)} datasets)'):
            if len(datapoint['images']) > 1:
                # llama 3.2 can't handle more than one image
                continue
            if i == 0:
                num_decimals = 1
            else:
                num_decimals = math.floor(math.log(i, 10)) + 1
            num_zeros = 6 - num_decimals
            img_path = img_root / ('0' * num_zeros + str(i) + '.png')
            datapoint['images'][0].save(img_path)
            question = datapoint['texts'][0]['user']
            answer = datapoint['texts'][0]['assistant']
            question_before = state.choice(range(2))
            if question_before:
                question_text = question + '\n' + IMG_TOKEN
            else:
                question_text = IMG_TOKEN + '\n' + question
            messages = [
                {
                    'content': question_text,
                    'role': 'user'
                },
                {
                    'content': answer,
                    'role': 'assistant'
                }
            ]
            datapoint_dict = {
                'messages': messages,
                'images': [str(img_path)],
                'source': datapoint['texts'][0]['source']
            }
            conversations_per_split[split].append(datapoint_dict)
    for split, conversations in conversations_per_split.items():
        with open(DATAROOT / f'cauldron_01_15_25_{split}.json', 'w') as f:
            json.dump(conversations, f, indent=2)
    print(f'Done writing dataset jsons!')

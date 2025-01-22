from datasets import load_dataset
from pathlib import Path
import json
import jsonlines
import math
import numpy as np
from tqdm import tqdm

DATAROOT = Path('/import/ml-sc-scratch1/jonathanl/repositories/EtashGuha-LLaMA-Factory/data')
SEED = 42
IMG_TOKEN = '<image>'

if __name__ == '__main__':
    DATAROOT.mkdir(exist_ok=True)
    metadata_dict = dict()
    ds = load_dataset('EtashGuha/JapaneseDocQA', cache_dir='/import/ml-sc-scratch3/jonathanl/cache')
    state = np.random.RandomState(seed=SEED)
    for split in ds.keys():
        img_root = DATAROOT / 'jdocqa_1_21_2025' / split
        img_root.mkdir(exist_ok=True, parents=True)
        conversations = []
        for i, datapoint in tqdm(enumerate(ds[split]), total=len(ds[split]), dynamic_ncols=True, desc=split):
            if i == 0:
                num_decimals = 1
            else:
                num_decimals = math.floor(math.log(i, 10)) + 1
            num_zeros = 5 - num_decimals
            img_path = img_root / ('0' * num_zeros + str(i) + '.png')
            datapoint['image'].save(img_path)
            messages = []
            question = datapoint['question']
            answer = datapoint['original_answer']
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
            conversations.append({'messages': messages, 'images': [str(img_path)]})
        with open(DATAROOT / f'jdocqa_01_21_2025_{split}.json', 'w') as f:
            json.dump(conversations, f, indent=2)
        print(f'Done writing dataset json!')
        

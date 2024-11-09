from datasets import load_dataset
from pathlib import Path
import json
import jsonlines
import math
import numpy as np
from tqdm import tqdm

DATAROOT = Path('/nvmedata/jonathanl/LLaMA-Factory/data')
SEED = 42
IMG_TOKEN = '<image>'

if __name__ == '__main__':
    DATAROOT.mkdir(exist_ok=True)
    metadata_dict = dict()
    ds = load_dataset('makcedward/hudocvqa', cache_dir='/nvmedata/jonathanl/hf_cache')
    img_root = DATAROOT / 'makcedward_hudocvqa_11_06_2024'
    img_root.mkdir(exist_ok=True)
    split = 'train'
    conversations = []
    state = np.random.RandomState(seed=SEED)
    for i, datapoint in tqdm(enumerate(ds['train']), total=len(ds['train']), dynamic_ncols=True, desc=f'train'):
        if i == 0:
            num_decimals = 1
        else:
            num_decimals = math.floor(math.log(i, 10)) + 1
        num_zeros = 5 - num_decimals
        img_path = img_root / ('0' * num_zeros + str(i) + '.png')
        datapoint['image'].save(img_path)
        messages = []
        num_qas = len(datapoint['questions'])
        qa_idx = state.choice(range(num_qas))
        question = datapoint['questions'][qa_idx]
        answer = datapoint['answers'][qa_idx]
        question_before = state.choice(range(2))
        messages = [
            {
                'content': IMG_TOKEN + '\n' + question,
                'role': 'user'
            },
            {
                'content': answer,
                'role': 'assistant'
            }
        ]
        conversations.append({'messages': messages, 'images': [str(img_path)]})
    with open(DATAROOT / 'makcedward_hudocvqa_11_06_2024.json', 'w') as f:
        json.dump(conversations, f, indent=2)
    print(f'Done writing dataset json!')

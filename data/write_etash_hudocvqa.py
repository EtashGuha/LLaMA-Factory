from datasets import load_dataset
from pathlib import Path
import json
import jsonlines
import math
import numpy as np
from tqdm import tqdm

SAVEROOT = Path('/import/ml-sc-scratch5/shubhangiu/etash_fork_multilingual/LLaMA-Factory/data')

SEED = 42
IMG_TOKEN = '<image>'

if __name__ == '__main__':
    metadata_dict = dict()
    gold_test_split = load_dataset('EtashGuha/HungarianDocQA', cache_dir="/import/ml-sc-scratch5/shubhangiu/cache/")
    total_test_samples = len(gold_test_split["test"])
    test_samples_split = int(total_test_samples/2)
    test_split = gold_test_split["test"].select(range(test_samples_split))
    train_split = gold_test_split["test"].select(range(test_samples_split, total_test_samples))
    state = np.random.RandomState(seed=SEED)
    split_to_dataset_mapping = {"train": train_split, "test": test_split}
    for split in ['train', 'test']:
        target_dataset = split_to_dataset_mapping[split]
        conversations = []
        img_root = SAVEROOT / 'etashg_hungarian_docQA' / split
        img_root.mkdir(exist_ok=True, parents=True)
        for i, datapoint in enumerate(target_dataset): 
            question = datapoint["Question"]
            answer = datapoint["Short Answer"] # short answer or answer??
            if i == 0:
                num_decimals = 1
            else:
                num_decimals = math.floor(math.log(i, 10)) + 1
            num_zeros = 5 - num_decimals
            img_path = img_root / ('0' * num_zeros + str(i) + '.png')
            datapoint['image'].save(img_path)
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

        with open(SAVEROOT / f'etashg_hungarian_docQA_12_04_2024_{split}.json', 'w') as f:
            json.dump(conversations, f, indent=2)
        print(f'Done writing dataset json!')

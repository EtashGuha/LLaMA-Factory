from datasets import load_dataset
from pathlib import Path
import io
import json
import jsonlines
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

DATAROOT = Path('/import/ml-sc-scratch1/jonathanl/repositories/EtashGuha-LLaMA-Factory/data')
SEED = 42
IMG_TOKEN = '<image>'

if __name__ == '__main__':
    DATAROOT.mkdir(exist_ok=True)
    metadata_dict = dict()
    ds = load_dataset('turing-motors/Cauldron-JA', streaming=True, cache_dir='/import/ml-sc-scratch3/jonathanl/cache')
    state = np.random.RandomState(seed=SEED)
    for split in ['train']:
        img_root = DATAROOT / 'cauldron_ja_01_13_2025' / split
        img_root.mkdir(exist_ok=True, parents=True)
        conversations = []
        try:
            total_length = len(ds[split])
        except:
            total_length = ds[split]._info.splits[split].num_examples
        pbar = tqdm(total=total_length, dynamic_ncols=True, desc=f'Cauldron-JA {split} split')
        i = 0
        ds_iterator = iter(ds[split])
        datapoint = next(ds_iterator)
        # skip to the next one
        # while i < 4341826:
        #     try:
        #         datapoint = next(ds_iterator)
        #     except:
        #         ds['train'].__dict__['_state_dict']['shard_idx'] += 1
        #         i += 1
        #         pbar.update(1)
        #         ds_iterator = iter(ds['train'])
        #         datapoint = next(ds_iterator)
        #     i += 1
        #     pbar.update(1)
        while datapoint is not None and i < total_length:
            if i == 0:
                num_decimals = 1
            else:
                num_decimals = math.floor(math.log(i, 10)) + 1
            num_zeros = 8 - num_decimals
            img_path = img_root / ('0' * num_zeros + str(i) + '.png')
            if len(datapoint['images']) > 1:
                # llama 3.2 cannot handle more than 1 image in the input
                i += 1
                pbar.update(1)
                datapoint = next(ds_iterator)
                continue
            image = Image.open(io.BytesIO(datapoint['images'][0]['bytes']))
            try:
                image.save(img_path)
            except:
                print(f'Failed to save image {i}, continuing!')
                try:
                    datapoint = next(ds_iterator)
                except:
                    ds['train'].__dict__['_state_dict']['shard_idx'] += 1
                    i += 1
                    pbar.update(1)
                    ds_iterator = iter(ds['train'])
                    datapoint = next(ds_iterator)
                i += 1
                pbar.update(1)
                continue
            messages = []
            num_qas = len(datapoint['texts'])
            qa_idx = state.choice(range(num_qas))
            question = datapoint['texts'][qa_idx]['jp_user']
            answer = datapoint['texts'][qa_idx]['jp_assistant']
            question_before = state.choice(range(2))
            question_text = IMG_TOKEN + '\n' + question
            # if question_before:
            #     question_text = question + '\n' + IMG_TOKEN
            # else:
            #     question_text = IMG_TOKEN + '\n' + question
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

            # iterate!
            try:
                datapoint = next(ds_iterator)
            except TypeError:
                print(f'Encountered TypeError with datapoint {i + 1}, skipping shard!', flush=True)
                ds['train'].__dict__['_state_dict']['shard_idx'] += 1
                i += 1
                pbar.update(1)
                ds_iterator = iter(ds['train'])
                datapoint = next(ds_iterator)
            i += 1
            pbar.update(1)
        pbar.close()
        with open(DATAROOT / f'caulrdon_ja_01_13_2025_{split}.json', 'w') as f:
            json.dump(conversations, f, indent=2)
        print(f'Done writing dataset json!', flush=True)

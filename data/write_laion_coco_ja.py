import json
import glob
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

SRCDIR = Path('/import/ml-sc-scratch5/vamsik/mm_ml_llms/datasets_dummy/')
IMG_TOKEN = '<image>'
SEED = 42

JP_CAPTION_PROMPTS = [
    "画像で起こっていることを説明してください。",
    "この写真で何が起こっているのか教えてください。",
    "画像に描かれているシーンを説明してください。",
    "ここで何が起こっているのか順を追って説明してください。",
    "画像に映っている出来事を詳しく説明してください。",
    "見えている状況を言葉で描写してください。",
    "写真の主な動きと背景（状況）を説明してください。",
    "この場面では何が起きていますか？",
    "この画像に捉えられた瞬間を物語っていただけますか？",
    "ここに描かれている状況を要約してください。",
    "映し出されている全体的な行動や出来事を説明してください。"
]

def read_src_json(filepath):
    with open(filepath) as f:
        ds_str = f.read()
    try:
        ds = json.loads('[' + ds_str[:-2] + ']')
    except:
        ds = json.loads('[' + ds_str[:-1] + ']')
    return ds

if __name__ == '__main__':
    state = np.random.RandomState(seed=SEED)
    source_jsons = glob.glob(str(SRCDIR / 'parallel_worker*train.json'))
    concatenated_list = []
    skipped = []
    for i, source_json in tqdm(enumerate(source_jsons), total=len(source_jsons), dynamic_ncols=True):
        ds = read_src_json(source_json)
        for j, datapoint in tqdm(enumerate(ds), total=len(ds), dynamic_ncols=True, desc=f'train file {i} / {len(source_jsons)}'):
            image_path = datapoint['images'][0]
            try:
                image = Image.open(image_path)
            except:
                skipped.append(image_path)
                print(f'Skipped entry {j} in file {source_json}', flush=True)
                continue
            question = state.choice(JP_CAPTION_PROMPTS)
            question_before = state.choice(range(2))
            if question_before == 0:
                question_text = question + '\n' + IMG_TOKEN
            else:
                question_text = IMG_TOKEN + '\n' + question
            assert datapoint['messages'][0]['content'] == ''
            datapoint['messages'][0]['content'] = question_text
            concatenated_list.append(datapoint)
    with open('laion_coco_ja_01_16_25_train.json', 'w') as f:
        json.dump(concatenated_list, f, indent=2)
    print('Done!')
    
    source_jsons = glob.glob(str(SRCDIR / 'parallel_worker*test.json'))
    concatenated_list = []
    skipped_test = []
    for source_json in tqdm(source_jsons, total=len(source_jsons), dynamic_ncols=True):
        ds = read_src_json(source_json)
        for j, datapoint in tqdm(enumerate(ds), total=len(ds), dynamic_ncols=True, desc=f'train file {i} / {len(source_jsons)}'):
            image_path = datapoint['images'][0]
            try:
                image = Image.open(image_path)
            except:
                skipped_test.append(image_path)
                print(f'Skipped entry {j} in file {source_json}', flush=True)
                continue
            question = state.choice(JP_CAPTION_PROMPTS)
            question_before = state.choice(range(2))
            if question_before == 0:
                question_text = question + '\n' + IMG_TOKEN
            else:
                question_text = IMG_TOKEN + '\n' + question
            assert datapoint['messages'][0]['content'] == ''
            datapoint['messages'][0]['content'] = question_text
            concatenated_list.append(datapoint)
    with open('laion_coco_ja_01_16_25_test.json', 'w') as f:
        json.dump(concatenated_list, f, indent=2)
    print('Done!')

    with open('laion_coco_ja_01_16_25_train.json') as f:
        ds_train = json.load(f)

    with open('laion_coco_ja_01_16_25_test.json') as f:
        ds_test = json.load(f)

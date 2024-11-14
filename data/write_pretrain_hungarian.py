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
DATASETS = [
    'hongfenglu/HungarianDocQA_IT_IFEvalQA',
    'nhiremath/RawHungarianPDFExtended_V2'
]

ENGLISH_DESCRIPTORS = ["attached", "provided", "given"]
ENGLISH_OBJECT = ["image", "document", "PDF", "pdf"]

ENGLISH_PROMPTS = [
    "Please transcribe the text in the <descriptor> <object>.",
    "Transcribe the content of the <descriptor> <object>.",
    "Can you write out the text in shown in the <descriptor> <object>?",
    "What is written in the <descriptor> <object>?",
    "Copy the contents of the <object> exactly as they appear.",
    "Convert the content of the <descriptor> <object> into text, please."
]

HUNGARIAN_PROMPTS = [
    "Írja le a csatolt kép tartalmát.",
    "Írja le a mellékelt dokumentum tartalmát.",
    "Írja le a kép tartalmát pontosan úgy, ahogy megjelenik.",
    "Írja le a dokumentum tartalmát pontosan úgy, ahogy megjelenik.",
    "Le tudná írni a képen látható szöveget?",
    "Tudná-e leírni a képen látható szöveget?",
    "Tudná-e leírni a megadott dokumentumból a szöveget?",
    "Kérem, alakítsa át a képet gépelt formátumba.",
    "Kérem, alakítsa át a dokumentum tartalmát szöveggé.",
    "Kérem, írja ki a megadott képen látható szöveget.",    
    "Kérjük, alakítsa át a dokumentumot gépelt formátummá.",
    "Másolja le a képről a szöveget írott formátumba.",
    "Másolja le a szöveget a dokumentumból írott formátumba.",
]

def construct_messages(datapoint, img_root, idx, rng):
    if i == 0:
        num_decimals = 1
    else:
        num_decimals = math.floor(math.log(i, 10)) + 1
    num_zeros = 5 - num_decimals
    img_path = img_root / ('0' * num_zeros + str(i) + '.png')
    datapoint['image'].save(img_path)
    messages = []
    english_prompt = rng.choice([0, 1])
    english_prompt = rng.choice([0, 1])
    if english_prompt == 1:
        desc_str = rng.choice(ENGLISH_DESCRIPTORS)
        obj_str = rng.choice(ENGLISH_OBJECT)
        prompt_str = rng.choice(ENGLISH_PROMPTS).replace('<descriptor>', desc_str).replace('<object>', obj_str)
    else:
        prompt_str = rng.choice(HUNGARIAN_PROMPTS)
    image_before = rng.choice([0, 1])
    if image_before:
        prompt_str = IMG_TOKEN + '\n' + prompt_str
    else:
        prompt_str = prompt_str + '\n' + IMG_TOKEN
    messages = [
        {
            'content': prompt_str,
            'role': 'user'
        },
        {
            'content': datapoint['text'],
            'role': 'assistant'
        }
    ]
    return messages, img_path

if __name__ == '__main__':
    DATAROOT.mkdir(parents=True, exist_ok=True)
    metadata_dict = dict()
    rng = np.random.RandomState(seed=SEED)
    train_conversations = []
    for split, dataset_name in zip(['train', 'test'], DATASETS):
        ds = load_dataset(dataset_name, split=split, cache_dir='/nvmedata/jonathanl/hf_cache')
        img_root = DATAROOT / dataset_name.split('/')[-1] / split
        img_root.mkdir(parents=True, exist_ok=True)
        idx_set = set()
        for i, datapoint in tqdm(enumerate(ds), total=len(ds), dynamic_ncols=True, desc=f'{dataset_name} {split} split'):
            if 'image_idx' in datapoint:
                i = datapoint['image_idx']
            if i in idx_set:
                continue
            else:
                idx_set.add(i)
            messages, img_path = construct_messages(datapoint, img_root, i, rng)
            train_conversations.append({'messages': messages, 'images': [str(img_path)], 'dataset_name': dataset_name})
    with open(DATAROOT / f'hungarian_pretrain_train_11_14_2024.json', 'w') as f:
        json.dump(train_conversations, f, indent=2)
    print(f'Done writing {split} dataset json!')
    
    val_conversations = []
    dataset_name = DATASETS[0]
    split = 'val'
    img_root = DATAROOT / dataset_name.split('/')[-1] / split
    img_root.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_name, split=split, cache_dir='/nvmedata/jonathanl/hf_cache')
    idx_set = set()
    for datapoint in tqdm(ds, total=len(ds), dynamic_ncols=True, desc=f'{dataset_name} {split} split'):
        if datapoint['image_idx'] in idx_set:
            continue
        else:
            idx_set.add(datapoint['image_idx'])
        message, img_path = construct_messages(datapoint, img_root, datapoint['image_idx'], rng)
        val_conversations.append({'messages': messages, 'images': [str(img_path)], 'dataset_name': dataset_name})
    with open(DATAROOT / f'hungarian_pretrain_{split}_11_14_2024.json', 'w') as f:
        json.dump(val_conversations, f, indent=2)
    print(f'Done writing {split} dataset json!')
    print('Done!')

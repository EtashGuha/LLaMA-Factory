import json
import glob
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers.image_utils import get_channel_dimension_axis, to_numpy_array
from transformers.models.mllama.image_processing_mllama import convert_to_rgb

FILE_TEMPLATE='/import/ml-sc-scratch1/jonathanl/repositories/EtashGuha-LLaMA-Factory/data/laion_coco_ja_01_16_25_{}.json'

if __name__ == '__main__':
    skipped_files = defaultdict(list)
    valid_datapoints = defaultdict(list)
    for split in ['test', 'train']:
        with open(FILE_TEMPLATE.format(split)) as f:
            ds = json.load(f)
        for i, datapoint in (pbar := tqdm(enumerate(ds), total=len(ds), dynamic_ncols=True, desc=f'{split}: 0 skipped')):
            image_path = datapoint['images'][0]
            image = Image.open(image_path)
            image = convert_to_rgb(image)
            try:
                img_array = to_numpy_array(image)
            except Exception as e:
                print(f'Skipped image {i} in split {split} due to exception: "{e}"')
                skipped_files[split].append(image_path)
                pbar.set_description(f'{split}: {len(skipped_files[split])} skipped')
                continue
            channel_dim = get_channel_dimension_axis(img_array)
            if img_array.shape[0] in (1,3) and img_array.shape[-1] in (1,3):
                # ambiguous!
                skipped_files[split].append(image_path)
                pbar.set_description(f'{split}: {len(skipped_files[split])} skipped')
                continue
            else:
                valid_datapoints[split].append(datapoint)
    
    # for split in ['test', 'train']:
    #     with open(f'laion_coco_ja_filtered_01_18_25_{split}.json', 'w') as f:
    #         json.dump(valid_datapoints[split], f, indent=2)

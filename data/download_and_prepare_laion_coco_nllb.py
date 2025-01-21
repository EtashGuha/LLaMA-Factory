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
IMG_TOKEN = '<image>'
DATAROOT = Path('/import/ml-sc-scratch5/vamsik/mm_ml_llms/datasets/liaon_coco_jap')
DATAROOT.mkdir(exist_ok=True)

print(f"HF_HOME: {os.getenv('HF_HOME')}")
print(f"HF_DATASETS_CACHE: {os.getenv('HF_DATASETS_CACHE')}")

os.environ["HF_HOME"] = "/import/ml-sc-scratch5/vamsik/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/import/ml-sc-scratch5/vamsik/hf_cache"

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

class HuggingFaceDatasetWrapper(Dataset):
    def __init__(self, dataset_name="visheratin/laion-coco-nllb", split="train", cache_dir="data_cache"):
        # Load the dataset from Hugging Face
        self.dataset = load_dataset(dataset_name, split=split, cache_dir="/import/ml-sc-scratch5/vamsik/hf_cache")
        self.split = split
        self.img_root = DATAROOT / self.split / 'images'
        self.img_root.mkdir(exist_ok=True, parents=True)
        self.json_annotation_root = DATAROOT / self.split / 'annotations' / 'worker_annotations'
        self.json_annotation_root.mkdir(exist_ok=True, parents=True)
        self.state = np.random.RandomState(seed=SEED)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract the relevant fields (adjust based on your dataset schema)
        row = self.dataset[idx]        

        data_id = row['id']
        image_url = row['url']

        worker_info = get_worker_info()
        worker_id = 0
        if worker_info is not None:
            worker_id = worker_info.id
        
        # Download the image.
        img_path = download_image_from_url(data_id, image_url, self.img_root)

        is_img_valid = validate_image(img_path)

        json_annotation_path = self.json_annotation_root / f'parallel_worker_{worker_id}_laion_coco_nllb_ja_01_21_2025_{self.split}.json'
        # download the caption, if the image is downloaded.
        if is_img_valid:
            english_caption = row['eng_caption']
            multi_lingual_captions_list = row['captions']
            for language, caption in multi_lingual_captions_list:
                if language == "jpn_Jpan":
                    question = self.state.choice(JP_CAPTION_PROMPTS)
                    question_before = self.state.choice(range(2))
                    if question_before == 0:
                        question_text = question + '\n' + IMG_TOKEN
                    else:
                        question_text = IMG_TOKEN + '\n' + question
                    messages = [
                        {
                            'content': question_text,
                            'role': 'user'
                        },
                        {
                            'content': caption,
                            'role': 'assistant'
                        }
                    ]
                    conversation = {'messages': messages, 'images': [str(img_path)], 'english_caption': english_caption, 'id': data_id}

                    with open(json_annotation_path, 'a') as f:
                        json.dump(conversation, f, indent=2)
                        f.write(",\n")

        return img_path is not None


def download_image_from_url(image_id, image_url, save_dir):
    img_path = save_dir / f'{image_id}.jpg'
    try:
        response = requests.get(image_url, timeout=2)
        if response.status_code == 200:
            with open(img_path, "wb") as file:
                file.write(response.content)
            return img_path
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}. Retrying...")

    return None


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


def collate_fn(batch):
    return batch


def read_src_json(filepath):
    with open(filepath) as f:
        ds_str = f.read()
    try:
        ds = json.loads('[' + ds_str[:-2] + ']')
    except:
        ds = json.loads('[' + ds_str[:-1] + ']')
    return ds

def merge_json_files(folder, split):

    worker_annotations_folder = folder / split / 'annotations' / 'worker_annotations'

    merged_json_data = []

    for filename in os.listdir(worker_annotations_folder):
        json_file_path = worker_annotations_folder / filename
        if filename.endswith('.json'):
            try:
                data = read_src_json(json_file_path)
                assert isinstance(data, list), "worker jsonfile is expected to be in list format."
                merged_json_data.extend(data)
            except Exception as e:
                print(f"Error reading {json_file_path}: {e}")
                return
    
    merged_json_file_path = folder / split / 'annotations' / 'merged_laion_coco_nllb_ja_01_21_2025_data.json'
    with open(merged_json_file_path, 'w') as outfile:
        json.dump(merged_json_data, outfile, indent=2)

    print(f"Merged JSON saved to {merged_json_file_path}")


def main():
    dataset_name = "visheratin/laion-coco-nllb"  # Replace with the Hugging Face dataset name
    splits = ["train", "test"]                    # Replace with the dataset split you want

    for split in splits:
        # Create the PyTorch dataset and DataLoader
        dataset = HuggingFaceDatasetWrapper(dataset_name, split)

        batch_size = 1
        num_workers = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

        total_num_examples = len(dataset)
        num_examples_processed = 0
        num_examples_downloaded = 0

        # Parallel processing with DataLoader
        for batch in dataloader:
            # Use PyTorch multiprocessing to parallelize the downloads
            num_examples_processed += batch_size
            for result in batch:
                if result:
                    num_examples_downloaded += 1
            print(f"num_examples_downloaded: {num_examples_downloaded} out of num_examples_processed: {num_examples_processed}")

        merge_json_files(DATAROOT, split)

if __name__ == "__main__":
    main()

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
import argparse
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import datasets
import pandas as pd
import os


def main(args):
    # load data
    print('loading datasets')
    test_df = load_chartqa_dataset("test")

    # main logic
    if args.task == 'blip_baseline':
        # loads InstructBLIP model
        print('loading models')
        model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=torch.device("cuda"))

        def image_gen(image_path, prompt="What is unusual about this image?", device=torch.device("cuda")):
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            return model.generate({"image": image, "prompt": prompt})
    
        correct_cnt = 0
        for idx, row in tqdm(test_df.iterrows()):
            imgname = row['imgname']
            query = row['query']
            label = row['label']
            image_path = f'ChartQA Dataset/test/png/{imgname}'
            generation = image_gen(image_path=image_path, prompt=query)
            if isinstance(generation, list):
                generation = generation[0]
            if generation == label:
                correct_cnt += 1
            if (idx+1) % 200 == 0 or idx == 9:
                print(f'idx: {idx}, accuracy: {correct_cnt / (idx+1) * 100:.2f}%')
        
        print(f'idx: {idx}, accuracy: {correct_cnt / (idx+1) * 100:.2f}%')


def load_chartqa_dataset(split, dataset_path='ChartQA Dataset/'):
    json_path_aug = os.path.join(dataset_path, split, f"{split}_augmented.json")
    json_path_human = os.path.join(dataset_path, split, f"{split}_human.json")
    df_aug = pd.read_json(json_path_aug)
    df_human = pd.read_json(json_path_human)
    df = pd.concat([df_aug, df_human])
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--task', type=str, help='task to run', choices=['blip_baseline'])

    # Parse the arguments
    args = parser.parse_args()
    main(args)
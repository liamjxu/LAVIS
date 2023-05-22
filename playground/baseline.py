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
        model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=torch.device("cuda"))

        def image_gen(image_path, prompt="What is unusual about this image?", device=torch.device("cuda")):
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            return model.generate({"image": image, "prompt": prompt})
    
        results = []
        correct_cnt = 0
        for idx, row in tqdm(test_df.iterrows()):
            imgname = row['imgname']
            query = text_wrap(row['query'], wrap=args.wrap)
            label = row['label']
            image_path = f'playground/ChartQA Dataset/test/png/{imgname}'
            generation = image_gen(image_path=image_path, prompt=query)
            if isinstance(generation, list):
                generation = generation[0]
            if generation == label:
                correct_cnt += 1
            result_entry = {
                'imgname': imgname,
                'query': query,
                'label': label,
                'generation': generation,
            }
            results.append(result_entry)
            if (idx+1) % 100 == 0 or idx == 9:
                print(f'idx: {idx}, accuracy: {correct_cnt / (idx+1) * 100:.2f}%')
                with open(f'playground/results/{args.output_filename}', 'w') as f:
                    json.dump(results, f, indent=4)

        print(f'idx: {idx}, accuracy: {correct_cnt / (idx+1) * 100:.2f}%')
        with open(f'playground/results/{args.output_filename}', 'w') as f:
            json.dump(results, f, indent=4)


def text_wrap(text, wrap='identity'):
    if wrap == 'identity':
        return text
    elif wrap == 'one_word':
        return f'Answer this question with one number or one phrase: {text}'
    elif wrap == 'zero_cot_one_word':
        return f"Think step-by-step and answer this question with only one number or one phrase: {text}"
    elif wrap == 'detect':
        return 'What symbolic elements are in this image? E.g., numbers, words, colors'


def load_chartqa_dataset(split, dataset_path='playground/ChartQA Dataset/'):
    json_path_aug = os.path.join(dataset_path, split, f"{split}_augmented.json")
    json_path_human = os.path.join(dataset_path, split, f"{split}_human.json")
    df_aug = pd.read_json(json_path_aug)
    df_human = pd.read_json(json_path_human)
    df = pd.concat([df_aug, df_human])
    df.reset_index(drop=True)
    print(f'The length of the resulting {split} df: {len(df)}')
    return df



if __name__ == '__main__':
    
    # set cache dir
    new_cache_dir = "/fs/scratch/rng_cr_rtc_hmi_gpu_user_c_lf/xji4syv/.cache"
    os.environ["TRANSFORMERS_CACHE"] = new_cache_dir

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--task', type=str, help='task to run', choices=['blip_baseline'])
    parser.add_argument('--model_name', type=str, help='the model name to use', choices=['blip2_t5_instruct'])
    parser.add_argument('--model_type', type=str, help='the model type to use', choices=['flant5xl', 'flant5xxl'])
    parser.add_argument('--output_filename', type=str, help='the filename to save output generations')
    parser.add_argument('--wrap', type=str, help='wrapping method of the query',
                        choices=['identity', 'one_word', 'detect', 'zero_cot_one_word'])

    # Parse the arguments
    args = parser.parse_args()
    main(args)
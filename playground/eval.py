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
    with open(f'playground/results/{args.result_filename}', 'r') as f:
        data = json.load(f)
    cnt = 0
    for entry in data:
        if entry['generation'] == entry['label']:
            cnt += 1
    print(f'Accuracy: {cnt/len(data)*100:.2f}%')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--result_filename', type=str, help='result to evaluate')

    # Parse the arguments
    args = parser.parse_args()
    main(args)
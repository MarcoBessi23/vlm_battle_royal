import argparse
import json
import re
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='transform results in pub tabnet format for evaluation')
    parser.add_argument('--input_path', 
                        required=True,
                        help='path to the results file',
                        type=str
    )
    parser.add_argument('--output_path',
                        required=True,
                        help='path to save results in the pubtabnet format',
                        type=str
    )
    return parser.parse_args()

def transform(input_path:str, output_path:str):
    '''
    function to extract the output path from the model results and transfom
    it in the html format specified in the PubTabNet dataset.
    '''
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("")
    with open(output_path, 'a') as f_out:
        for d in data:
            out = d['output']
            structure_tokens = re.findall(r"</?\w+>", out)
            cell_texts = re.findall(r">(.*?)<", out)
            html_like = {
                "cells": [{"tokens": list(text)} for text in cell_texts],
                "structure": {"tokens": structure_tokens},
            }
            new_item = {"filename":d["filename"], "html": html_like}
            f_out.write(json.dumps(new_item) + "\n")

def main():
    args = parse_args()
    transform(args.input_path, args.output_path)

if __name__ == '__main__':
    main()

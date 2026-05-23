import json
import torch
import numpy as np
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_model(model_path, device="cuda:0"):
    if not os.path.exists(os.path.join(model_path, "config.json")):
        # Try as HuggingFace model ID
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    return model, tokenizer


def calculate_conditional_ppl(prompt, output, model, tokenizer, device):
    full_text = prompt + output

    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    if input_ids.shape[1] <= prompt_len:
        return None

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        nll = outputs.loss

    return torch.exp(nll).item()


def main():
    parser = argparse.ArgumentParser(description="IPMark perplexity calculation")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--data_file", type=str, default="output.jsonl",
                        help="JSONL file with prompt and output_with_watermark fields")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. cuda:0)")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    model, tokenizer = load_model(args.model_path, args.device)

    ppl_scores = []

    with open(args.data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        try:
            item = json.loads(line)
            prompt = item['prompt']
            output = item['output_with_watermark']

            ppl = calculate_conditional_ppl(prompt, output, model, tokenizer, args.device)

            if ppl is not None:
                ppl_scores.append(ppl)

        except (json.JSONDecodeError, KeyError):
            pass

    if ppl_scores:
        avg_ppl = np.mean(ppl_scores)
        median_ppl = np.median(ppl_scores)
        print(f"Count: {len(ppl_scores)}")
        print(f"Mean PPL: {avg_ppl:.4f}")
        print(f"Median PPL: {median_ppl:.4f}")
    else:
        print("No valid samples found.")


if __name__ == "__main__":
    main()

import sys
import os
import json
import torch
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

markllm_root = os.environ.get("MARKLLM_ROOT", "../MarkLLM")
sys.path.append(markllm_root)

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig


class Config:
    model_path = os.environ.get("MODEL_PATH", "facebook/opt-1.3b")
    algorithm_name = os.environ.get("BASELINE_ALGORITHM", "KGW")
    algorithm_config = os.environ.get(
        "BASELINE_CONFIG",
        os.path.join(markllm_root, "config", "KGW.json"),
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    file_path = os.environ.get("DATA_PATH", "data/sample_c4.jsonl")
    save_path = os.environ.get("SAVE_PATH", "output_kgw.jsonl")
    cache_dir = os.environ.get("HF_HOME", None)
    markllm_root = markllm_root

    dataset_begin = 0
    dataset_end = 1000

    TARGET_SENTENCE_COUNT = 20


PUNCT_TOKEN_RE = re.compile(r'^\s*(?:\n|\.{3}|[.?!，,。？！…]+(?:\s)?)\s*$')


def count_sentences(text):
    count = len(re.findall(r'[.?!。？！]', text))
    return count


def clean_text_to_exact_sentences(text, target_count):
    end_indices = [m.start() for m in re.finditer(r'[.?!。？！]', text)]

    if len(end_indices) >= target_count:
        cut_index = end_indices[target_count - 1] + 1
        return text[:cut_index]
    else:
        return text


def main():
    args = Config()
    original_cwd = os.getcwd()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, cache_dir=args.cache_dir, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, cache_dir=args.cache_dir,
            device_map=args.device, torch_dtype=torch.float16, trust_remote_code=True
        )

        if hasattr(model, "lm_head"):
            real_vocab_size = model.lm_head.out_features
        elif hasattr(model, "get_output_embeddings"):
            real_vocab_size = model.get_output_embeddings().weight.shape[0]
        else:
            real_vocab_size = 128000

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    transformers_config = TransformersConfig(
        model=model, tokenizer=tokenizer, vocab_size=real_vocab_size, device=args.device
    )

    try:
        os.chdir(args.markllm_root)
        load_kwargs = {"transformers_config": transformers_config}
        if os.path.exists(args.algorithm_config):
            load_kwargs["algorithm_config"] = args.algorithm_config

        pipeline = AutoWatermark.load(args.algorithm_name, **load_kwargs)
        os.chdir(original_cwd)
    except Exception as e:
        os.chdir(original_cwd)
        raise RuntimeError(f"Failed to load MarkLLM pipeline: {e}")

    if not os.path.exists(args.file_path):
        raise FileNotFoundError(f"Data file not found: {args.file_path}")

    with open(args.file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file if line.strip()]

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    mode = 'a' if os.path.exists(args.save_path) else 'w'

    with open(args.save_path, mode, encoding='utf-8') as outfile:
        end_idx = min(args.dataset_end, len(data))

        for i in range(args.dataset_begin, end_idx):
            prompt_text = data[i]['prompt']

            try:
                current_text = ""
                full_input = prompt_text

                loop_limit = 10

                for loop in range(loop_limit):
                    current_sent_count = count_sentences(current_text)
                    if current_sent_count >= args.TARGET_SENTENCE_COUNT:
                        break

                    step_output = pipeline.generate_watermarked_text(
                        full_input,
                        max_new_tokens=60,
                        min_new_tokens=10,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.75,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                    )

                    if step_output.startswith(full_input):
                        new_part = step_output[len(full_input):]
                    else:
                        new_part = step_output

                    if not new_part.strip():
                        new_part = " "

                    current_text += new_part
                    full_input += new_part

                    if len(current_text) > 2000:
                        break

                final_output = clean_text_to_exact_sentences(
                    current_text, args.TARGET_SENTENCE_COUNT
                )

                result_entry = {
                    "test_index": i,
                    "prompt": prompt_text,
                    "output_with_watermark": final_output,
                    "sentence_count": count_sentences(final_output),
                }

                outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                outfile.flush()

            except Exception as e:
                import traceback
                traceback.print_exc()

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

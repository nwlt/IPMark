import os
import json
import torch
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Config:
    model_path = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.2-3B")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    file_path = os.environ.get("DATA_PATH", "data/sample_c4.jsonl")
    save_path = os.environ.get("SAVE_PATH", "output_no_wm.jsonl")
    cache_dir = os.environ.get("HF_HOME", None)

    dataset_begin = 0
    dataset_end = 1000

    TARGET_SENTENCE_COUNT = 20


def count_sentences(text):
    return len(re.findall(r'[.?!。？！]', text))


def clean_text_to_exact_sentences(text, target_count):
    end_indices = [m.start() for m in re.finditer(r'[.?!。？！]', text)]
    if len(end_indices) >= target_count:
        cut_index = end_indices[target_count - 1] + 1
        return text[:cut_index]
    else:
        return text


@torch.no_grad()
def main():
    args = Config()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, cache_dir=args.cache_dir, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(args.device)
        model.eval()

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

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
                    if count_sentences(current_text) >= args.TARGET_SENTENCE_COUNT:
                        break

                    enc = tokenizer(full_input, return_tensors="pt", padding=False)
                    input_ids = enc.input_ids.to(args.device)
                    attn_mask = enc.attention_mask.to(args.device)

                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=60,
                        min_new_tokens=10,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.75,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    new_ids = gen_ids[0, input_ids.shape[1]:]
                    new_part = tokenizer.decode(new_ids, skip_special_tokens=True)

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

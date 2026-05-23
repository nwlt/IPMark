import os
import sys
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from tqdm import tqdm

hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Config:
    markllm_root = os.environ.get("MARKLLM_ROOT", "../MarkLLM")
    algorithm_name = os.environ.get("BASELINE_ALGORITHM", "Unigram")
    algorithm_config = os.environ.get(
        "BASELINE_CONFIG",
        os.path.join(markllm_root, "config", "Unigram.json"),
    )
    model_path = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B")
    cache_dir = os.environ.get("HF_HOME", None)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    input_jsonl = os.environ.get("INPUT_JSONL", "attacked.jsonl")
    output_jsonl = os.environ.get("OUTPUT_JSONL", "attacked_detect_output.jsonl")

    index_field = "test_index"
    original_field = "output_with_watermark"
    attacked_field = "attacked_versions"


def init_markllm(markllm_root: str):
    sys.path.append(markllm_root)
    from watermark.auto_watermark import AutoWatermark
    from utils.transformers_config import TransformersConfig
    return AutoWatermark, TransformersConfig


def extract_score(detect_result):
    z = None
    if isinstance(detect_result, dict):
        z = (
            detect_result.get("z_score")
            or detect_result.get("score")
            or detect_result.get("z")
            or detect_result.get("p_value")
        )
    elif isinstance(detect_result, (tuple, list)) and len(detect_result) >= 2:
        z = detect_result[1]

    if z is None:
        raise ValueError(f"Score field not found in detect_watermark return: {detect_result}")
    return float(z)


@torch.no_grad()
def main():
    args = Config()
    AutoWatermark, TransformersConfig = init_markllm(args.markllm_root)

    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        real_vocab_size = model.get_output_embeddings().weight.shape[0]
    else:
        real_vocab_size = len(tokenizer)

    transformers_config = TransformersConfig(
        model=model, tokenizer=tokenizer,
        vocab_size=real_vocab_size, device=args.device,
    )
    watermark = AutoWatermark.load(
        args.algorithm_name,
        algorithm_config=args.algorithm_config,
        transformers_config=transformers_config,
    )

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output_jsonl, "w", encoding="utf-8") as fout:

        pbar = tqdm(fin, total=total_lines, desc="Detecting", unit="line")

        for line_no, line in enumerate(pbar, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                test_idx = obj.get(args.index_field, line_no)
                results = {}

                if args.original_field in obj and obj[args.original_field]:
                    det = watermark.detect_watermark(obj[args.original_field])
                    results["Original"] = {
                        "test_index": test_idx,
                        "z_total_rate": extract_score(det),
                        "is_watermarked": (
                            bool(det.get("is_watermarked")) if isinstance(det, dict) else None
                        ),
                    }

                attacked = obj.get(args.attacked_field, {}) or {}
                for attack_name, attacked_text in attacked.items():
                    if not attacked_text or str(attacked_text).strip() == "":
                        results[attack_name] = {
                            "test_index": test_idx,
                            "z_total_rate": None,
                            "is_watermarked": None,
                        }
                        continue

                    det = watermark.detect_watermark(attacked_text)
                    results[attack_name] = {
                        "test_index": test_idx,
                        "z_total_rate": extract_score(det),
                        "is_watermarked": (
                            bool(det.get("is_watermarked")) if isinstance(det, dict) else None
                        ),
                    }

                out_line = {"test_index": test_idx, "results": results}
                fout.write(json.dumps(out_line, ensure_ascii=False) + "\n")

                pbar.set_postfix({"idx": test_idx})

            except Exception as e:
                pbar.write(f"[Error] line {line_no}: {e}")

            if line_no % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

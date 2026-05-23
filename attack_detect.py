import json
import argparse
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from watermark_engine import generate_watermark_keys
from utils import split_sentences, filter_short_word_sentences, TRANSLATOR

p0 = 0.5


def process_single_text(text, test_index_val, syntax_weight=0.5):
    sentences = split_sentences(text)
    sentences = filter_short_word_sentences(sentences)

    model_id_count_valid = 0
    model_id_user_id_count_valid = 0
    model_id_count_total = 0
    model_id_user_id_count_total = 0

    for sent in sentences:
        sentence_withoutpunctuation = sent.translate(TRANSLATOR).strip()
        if not sentence_withoutpunctuation:
            continue

        key_type, HMAC_syntax, HMAC_semanteme = generate_watermark_keys(
            sentence_withoutpunctuation
        )
        semanteme_w = 1.0 - syntax_weight
        score = (HMAC_semanteme % 2) * semanteme_w + (HMAC_syntax % 2) * syntax_weight

        if key_type == "model_id":
            model_id_count_total += 1
            model_id_count_valid += score
        else:
            model_id_user_id_count_total += 1
            model_id_user_id_count_valid += score

    result = {
        "test_index": test_index_val,
        "z_model_id_rate": (
            (model_id_count_valid * (20 / model_id_count_total) - p0 * 20)
            / np.sqrt(p0 * (1 - p0) * 20)
        ) if model_id_count_total > 0 else None,
        "z_model_id.user_id_rate": (
            (model_id_user_id_count_valid * (20 / model_id_user_id_count_total) - p0 * 20)
            / np.sqrt(p0 * (1 - p0) * 20)
        ) if model_id_user_id_count_total > 0 else None,
        "z_total_rate": (
            ((model_id_count_valid + model_id_user_id_count_valid)
             * (20 / (model_id_user_id_count_total + model_id_count_total)) - p0 * 20)
            / np.sqrt(p0 * (1 - p0) * 20)
        ) if (model_id_user_id_count_total > 0) or (model_id_count_total > 0) else None,
    }

    if (result["z_model_id_rate"] is None
            or result["z_model_id.user_id_rate"] is None
            or len(sentences) < 4):
        result["too_few_sentences"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description="IPMark attack detection")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSONL file with attacked watermarked texts")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file for detection results")
    parser.add_argument("--syntax_weight", type=float, default=0.5,
                        help="Weight for syntax score")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    stats = defaultdict(list)

    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout:
        for i, line in enumerate(tqdm(fin, desc="Processing lines")):
            entry = json.loads(line)
            test_idx = entry.get("test_index", i)
            tasks = {}

            if entry.get("output_with_watermark"):
                tasks["Original"] = entry.get("output_with_watermark")

            attacks = entry.get("attacked_versions", {})
            if attacks:
                tasks.update(attacks)

            entry_robustness = {"test_index": test_idx, "results": {}}

            for name, text in tasks.items():
                res = process_single_text(text, test_idx, args.syntax_weight)
                entry_robustness["results"][name] = res

                if res.get("z_total_rate") is not None:
                    stats[name].append(res["z_total_rate"])

            fout.write(json.dumps(entry_robustness, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

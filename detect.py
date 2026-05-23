import json
import re
import numpy as np
import argparse
import os
from tqdm import tqdm
from watermark_engine import generate_watermark_keys
from utils import split_sentences, filter_short_word_sentences, TRANSLATOR

p0 = 0.5


def main():
    parser = argparse.ArgumentParser(description="IPMark watermark detection")
    parser.add_argument("--input_file", type=str, default="output.jsonl")
    parser.add_argument("--output_file", type=str, default="detect_result.jsonl")
    parser.add_argument("--syntax_weight", type=float, default=0.5,
                        help="Weight for syntax score (semanteme weight = 1 - syntax_weight)")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout:
        for i, line in enumerate(tqdm(fin)):
            entry = json.loads(line)
            text = entry.get("output_with_watermark")

            sentences = split_sentences(text)
            sentences = filter_short_word_sentences(sentences)

            model_id_count_valid = 0
            model_id_user_id_count_valid = 0
            model_id_count_total = 0
            model_id_user_id_count_total = 0

            for sent in sentences:
                sentence_withoutpunctuation = sent.translate(TRANSLATOR).strip()
                key_type, HMAC_syntax, HMAC_semanteme = generate_watermark_keys(
                    sentence_withoutpunctuation
                )

                syntax_w = args.syntax_weight
                semanteme_w = 1.0 - args.syntax_weight
                score = (HMAC_semanteme % 2) * semanteme_w + (HMAC_syntax % 2) * syntax_w

                if key_type == "model_id":
                    model_id_count_total += 1
                    model_id_count_valid += score
                else:
                    model_id_user_id_count_total += 1
                    model_id_user_id_count_valid += score

            result = {
                "test_index": entry.get("test_index", i),
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

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

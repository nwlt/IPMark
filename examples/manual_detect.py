#!/usr/bin/env python3
"""Manual watermark detection example with detailed per-sentence output.

Note: This example uses 0.7/0.3 semanteme/syntax weights,
which differ from the 0.5/0.5 default in detect.py.
These weights were chosen based on empirical tuning.
"""

import json
import numpy as np
from watermark_engine import generate_watermark_keys
from utils import split_sentences, filter_short_word_sentences, TRANSLATOR

p0 = 0.5

# Example text (replace with your own watermarked output)
input_text = (
    "I'm looking forward to seeing you all again. "
    "Look at them. "
    "I'm sure you've never heard of him. "
    "He's from Punch-Out ! ! "
    "He was a man of courage and a wit of no small talent. "
    "I think he'll be a useful fighter. "
    "A few years back, a new breed of pranksters started hitting the streets. "
    "Duck Hunt is a new character that's going to be appearing in the game. "
    "Duck Hunt is a hunting dog that's capable of catching a duck or a goose. "
    "But, as you can see in the following trailer, he's pretty powerful, "
    "especially with his staff. "
    "It's not only his first appearance in a Nintendo game; he's also a playable character. "
    "It was the turn of the nymph, a very beautiful one. "
    "In the following clip, we see her in action, driving her car with a sledgehammer. "
    "I'm just a man who doesn't understand. "
    "These are the new characters that have been added to the roster."
)

sentences = split_sentences(input_text)
sentences = filter_short_word_sentences(sentences)

model_id_count_valid = 0
model_id_user_id_count_valid = 0
model_id_count_total = 0
model_id_user_id_count_total = 0

detailed_results = []

for i, sent in enumerate(sentences):
    sentence_withoutpunctuation = sent.translate(TRANSLATOR).strip()

    key_type, HMAC_syntax, HMAC_semanteme = generate_watermark_keys(
        sentence_withoutpunctuation
    )

    # Empirically tuned weights (0.7 semanteme, 0.3 syntax)
    score = (HMAC_semanteme % 2) * 0.7 + (HMAC_syntax % 2) * 0.3

    if key_type == "model_id":
        source_label = "model_id"
        model_id_count_total += 1
        model_id_count_valid += score
    else:
        source_label = "user_id"
        model_id_user_id_count_total += 1
        model_id_user_id_count_valid += score

    detailed_results.append({
        "index": i + 1,
        "sentence": sent.strip(),
        "detected_source": source_label,
        "raw_key_type": key_type,
        "score": score,
    })

result = {
    "model_id_score": (
        (model_id_count_valid * (20 / model_id_count_total) - p0 * 20)
        / np.sqrt(p0 * (1 - p0) * 20)
    ) if model_id_count_total > 0 else None,
    "model_id.user_id_score": (
        (model_id_user_id_count_valid * (20 / model_id_user_id_count_total) - p0 * 20)
        / np.sqrt(p0 * (1 - p0) * 20)
    ) if model_id_user_id_count_total > 0 else None,
    "total_sentences": len(sentences),
    "details": detailed_results,
}

print(json.dumps(result, indent=2, ensure_ascii=False))

#!/usr/bin/env python3
"""Detection speed benchmark with optional multiprocessing.

Note: This example uses 0.7/0.3 semanteme/syntax weights,
which differ from the 0.5/0.5 default in detect.py.
"""

import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from watermark_engine import generate_watermark_keys
from utils import split_sentences, filter_short_word_sentences, TRANSLATOR


def worker(cleaned: str):
    key_type, HMAC_syntax, HMAC_semanteme = generate_watermark_keys(cleaned)
    score = (HMAC_semanteme % 2) * 0.7 + (HMAC_syntax % 2) * 0.3
    return key_type, score


def can_use_fork():
    try:
        return "fork" in mp.get_all_start_methods()
    except Exception:
        return False


# Example text for benchmark
input_text = (
    " saying the United States would default on its debt if lawmakers failed "
    "to raise the debt ceiling.\n"
    "Lew, in an interview on NBC's \"Meet the Press,\" said the United States "
    "would default on its debt if lawmakers failed to raise the debt ceiling. "
    "The U.S. Treasury has said it will run out of money to pay all its bills "
    "by Oct. 17 if Congress does not act.\n"
    "\"If we don't raise the debt ceiling, we will default on our debt. "
    "It's that simple,\""
)

sentences = split_sentences(input_text)
sentences = filter_short_word_sentences(sentences)

prepared = [s.translate(TRANSLATOR).strip() for s in sentences]

model_id_count_valid = 0.0
model_id_user_id_count_valid = 0.0
model_id_count_total = 0
model_id_user_id_count_total = 0

use_fork_pool = can_use_fork() and len(prepared) >= 200
chunksize = 32

generate_watermark_keys("warm up for spacy and wordnet")

t0_all = time.perf_counter()

if use_fork_pool:
    ctx = mp.get_context("fork")

    with ProcessPoolExecutor(mp_context=ctx) as ex:
        t0_det = time.perf_counter()
        for key_type, score in ex.map(worker, prepared, chunksize=chunksize):
            if key_type == "model_id":
                model_id_count_total += 1
                model_id_count_valid += score
            else:
                model_id_user_id_count_total += 1
                model_id_user_id_count_valid += score
        t1_det = time.perf_counter()

    detect_time_total = t1_det - t0_det

else:
    t0_det = time.perf_counter()
    for cleaned in prepared:
        key_type, HMAC_syntax, HMAC_semanteme = generate_watermark_keys(cleaned)
        score = (HMAC_semanteme % 2) * 0.7 + (HMAC_syntax % 2) * 0.3

        if key_type == "model_id":
            model_id_count_total += 1
            model_id_count_valid += score
        else:
            model_id_user_id_count_total += 1
            model_id_user_id_count_valid += score
    t1_det = time.perf_counter()
    detect_time_total = t1_det - t0_det

t1_all = time.perf_counter()
pipeline_time_total = t1_all - t0_all

avg_detect_ms = (detect_time_total / len(prepared) * 1000) if prepared else 0.0
speed_sps = (len(prepared) / detect_time_total) if detect_time_total > 0 else None

result = {
    "model_id_rate": (
        (model_id_count_valid / model_id_count_total) if model_id_count_total > 0 else None
    ),
    "model_id.user_id_rate": (
        (model_id_user_id_count_valid / model_id_user_id_count_total)
        if model_id_user_id_count_total > 0 else None
    ),
    "total_sentences": len(sentences),
    "mode": "fork_process_pool" if use_fork_pool else "single_process",
    "detect_time_total_sec": detect_time_total,
    "detect_time_avg_ms": avg_detect_ms,
    "detect_speed_sentences_per_sec": speed_sps,
    "pipeline_time_total_sec": pipeline_time_total,
}

print(json.dumps(result, indent=2, ensure_ascii=False))

import os
import argparse
from generate import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMark: Sentence-Level LLM Watermarking")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--load_fp16", action="store_true")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--gpu", type=str, default="cuda:0")
    parser.add_argument("--prompt_max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_position_embeddings", type=int, default=2048)
    parser.add_argument("--generation_seed", type=int, default=123)
    parser.add_argument("--sampling_temp", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--accepted_sentences_target", type=int, default=20)
    parser.add_argument("--skip_model_load", action="store_true")
    parser.add_argument("--file_path", type=str, default="data/sample_c4.jsonl")
    parser.add_argument("--save_path", type=str, default="output.jsonl")
    parser.add_argument("--dataset_begin", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=1000)
    parser.add_argument("--normalizers", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--compile_model", action="store_true")

    args = parser.parse_args()
    main(args)

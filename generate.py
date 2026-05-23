import os
import argparse
import torch
import gc
import json

import torch.nn.functional as F

from watermark_engine import generate_watermark_keys
from utils import load_model, reorder_cache_manual, TRANSLATOR, is_punct_token
from typing import List, Dict, Any


def generate(prompt, args, model=None, device=None, tokenizer=None):
    tokd_input = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.prompt_max_length,
    ).to(device)
    input_ids = tokd_input["input_ids"]

    accepted_tokens: List[int] = []

    model.eval()
    torch.manual_seed(args.generation_seed)

    beam_width = args.beam_width
    accepted_sentences_target = args.accepted_sentences_target

    def default_selection_fn(candidates: List[Dict[str, Any]], wm_state: Dict[str, int], wm_thr: int):
        recent_keys = wm_state.setdefault("recent_wm_keys", [])

        def select_from_candidate_list(candidate_indices):
            if not candidate_indices:
                return None, False

            length_penalty_alpha = 0.7

            def get_norm_score(idx):
                c = candidates[idx]
                length = len(c["token_ids"])
                return c["logprob"] / ((length + 1e-6) ** length_penalty_alpha)

            sorted_by_score = sorted(candidate_indices, key=get_norm_score, reverse=True)
            best_idx = sorted_by_score[0]
            best_key = candidates[best_idx]["wm_key"]

            if len(recent_keys) >= 3 and all(key == best_key for key in recent_keys[-3:]):
                for idx in sorted_by_score:
                    if candidates[idx]["wm_key"] != best_key:
                        return idx, False
                return best_idx, True
            else:
                return best_idx, False

        wm_ok_both = [
            i for i, c in enumerate(candidates)
            if (c["HMAC_syntax"] % 2 == 1 and c["HMAC_semanteme"] % 2 == 1)
        ]
        if wm_ok_both:
            chosen_idx, _ = select_from_candidate_list(wm_ok_both)
            if chosen_idx is not None:
                recent_keys.append(candidates[chosen_idx]["wm_key"])
                if len(recent_keys) > 3:
                    recent_keys.pop(0)
                return chosen_idx

        wm_ok_semanteme = [i for i, c in enumerate(candidates) if (c["HMAC_semanteme"] % 2 == 1)]
        if wm_ok_semanteme:
            chosen_idx, _ = select_from_candidate_list(wm_ok_semanteme)
            if chosen_idx is not None:
                recent_keys.append(candidates[chosen_idx]["wm_key"])
                if len(recent_keys) > 3:
                    recent_keys.pop(0)
                return chosen_idx

        wm_ok_syntax = [i for i, c in enumerate(candidates) if (c["HMAC_syntax"] % 2 == 1)]
        if wm_ok_syntax:
            chosen_idx, _ = select_from_candidate_list(wm_ok_syntax)
            if chosen_idx is not None:
                recent_keys.append(candidates[chosen_idx]["wm_key"])
                if len(recent_keys) > 3:
                    recent_keys.pop(0)
                return chosen_idx

        all_indices = list(range(len(candidates)))
        chosen_idx, _ = select_from_candidate_list(all_indices)
        if chosen_idx is not None:
            recent_keys.append(candidates[chosen_idx]["wm_key"])
            if len(recent_keys) > 3:
                recent_keys.pop(0)
            return chosen_idx
        return 0

    def beam_search_generate_sentences(
        input_ids, accepted_tokens, beam_width=8, max_new_tokens=100, temperature=0.7
    ) -> List[Dict[str, Any]]:
        max_context_window = args.max_position_embeddings
        safe_window = max_context_window - max_new_tokens - 50

        acc_tensor = torch.tensor([accepted_tokens], device=device, dtype=input_ids.dtype)
        full_context = torch.cat([input_ids, acc_tensor], dim=1)

        if full_context.shape[1] > safe_window:
            full_context = full_context[:, -safe_window:]

        input_ids_batch = full_context.repeat(beam_width, 1)

        beam_scores = torch.zeros((beam_width,), device=device)
        beam_scores[1:] = -1e9

        beam_sequences = [[] for _ in range(beam_width)]
        past_key_values = None
        completed_candidates = []

        vocab_size = model.config.vocab_size

        try:
            for step in range(max_new_tokens):
                if len(completed_candidates) >= beam_width:
                    break

                if past_key_values is None:
                    model_inputs = input_ids_batch
                else:
                    model_inputs = input_ids_batch[:, -1:]

                with torch.no_grad():
                    outputs = model(
                        input_ids=model_inputs,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                repetition_penalty = 1.2
                current_context = input_ids_batch
                score = torch.gather(next_token_logits, 1, current_context)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                next_token_logits.scatter_(1, current_context, score)

                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                next_token_scores = F.log_softmax(next_token_logits, dim=-1)

                candidate_scores = beam_scores.unsqueeze(-1) + next_token_scores
                candidate_scores_flat = candidate_scores.view(-1)

                topk_scores, topk_indices = torch.topk(candidate_scores_flat, k=beam_width * 2)

                beam_indices = topk_indices // vocab_size
                token_indices = topk_indices % vocab_size

                next_beam_scores = []
                next_beam_sequences = []
                next_beam_indices_for_cache = []
                next_input_ids_list = []

                cnt_active = 0
                for i in range(len(topk_scores)):
                    if cnt_active >= beam_width:
                        break

                    bid = beam_indices[i].item()
                    tid = token_indices[i].item()
                    score = topk_scores[i].item()

                    token_str = tokenizer.decode([tid], skip_special_tokens=True)

                    if is_punct_token(token_str):
                        full_seq_ids = beam_sequences[bid] + [tid]
                        full_text = tokenizer.decode(full_seq_ids, skip_special_tokens=True)

                        sentence_clean = full_text.translate(TRANSLATOR).strip()
                        Key_type, HMAC_syntax, HMAC_semanteme = generate_watermark_keys(sentence_clean)

                        candidate = {
                            "token_ids": full_seq_ids,
                            "text": full_text,
                            "logprob": score,
                            "wm_key": Key_type,
                            "HMAC_syntax": HMAC_syntax,
                            "HMAC_semanteme": HMAC_semanteme,
                        }
                        completed_candidates.append(candidate)
                    else:
                        next_beam_scores.append(score)
                        next_beam_sequences.append(beam_sequences[bid] + [tid])
                        next_beam_indices_for_cache.append(bid)
                        next_input_ids_list.append(tid)
                        cnt_active += 1

                if len(next_beam_indices_for_cache) == 0:
                    break

                beam_scores = torch.tensor(next_beam_scores, device=device)
                beam_sequences = next_beam_sequences
                input_ids_batch = torch.tensor(next_input_ids_list, device=device).unsqueeze(1)

                if past_key_values is not None:
                    idx_tensor = torch.tensor(next_beam_indices_for_cache, device=device)
                    past_key_values = reorder_cache_manual(past_key_values, idx_tensor)

            length_penalty_alpha = 0.7

            def normalized_score(candidate):
                length = len(candidate["token_ids"])
                return candidate["logprob"] / ((length + 1e-6) ** length_penalty_alpha)

            completed_candidates.sort(key=normalized_score, reverse=True)

            return completed_candidates[:beam_width]

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    accepted_sentences = 0
    wm_state = {}
    temperature = args.sampling_temp
    max_new_tokens = args.max_new_tokens

    while accepted_sentences < accepted_sentences_target:
        candidates = beam_search_generate_sentences(
            input_ids, accepted_tokens,
            beam_width=beam_width, max_new_tokens=max_new_tokens, temperature=temperature
        )

        if not candidates:
            break

        chosen_idx = default_selection_fn(candidates, wm_state, None)
        chosen = candidates[chosen_idx]

        accepted_tokens.extend(chosen["token_ids"])

        if len(chosen["token_ids"]) > 3:
            accepted_sentences += 1
        else:
            accepted_sentences += 0.2
        torch.cuda.empty_cache()

    final_output_ids = (
        torch.tensor(accepted_tokens, device=device).unsqueeze(0)
        if len(accepted_tokens) > 0
        else torch.zeros((1, 0), dtype=torch.long, device=device)
    )
    decoded_output_with_watermark = tokenizer.decode(final_output_ids[0], skip_special_tokens=True)
    redecoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return redecoded_input, 0, decoded_output_with_watermark, args


def main(args):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    cache_dir = os.environ.get("HF_HOME", None)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args, cache_dir)
    else:
        model, tokenizer, device = None, None, None

    file_path = args.file_path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file if line.strip()]

    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as file:
        end_idx = min(args.dataset_end, len(data))

        for i in range(args.dataset_begin, end_idx):
            input_text = data[i]['prompt']

            if not args.skip_model_load:
                args.default_prompt = input_text

                _, _, decoded_output_with_watermark, _ = generate(
                    input_text,
                    args,
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                )

                result_entry = {
                    "test_index": i,
                    "prompt": input_text,
                    "output_with_watermark": decoded_output_with_watermark,
                }

                file.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                file.flush()

                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--beam_width", type=int, default=8)
    parser.add_argument("--accepted_sentences_target", type=int, default=20)
    parser.add_argument("--skip_model_load", action="store_true")
    parser.add_argument("--file_path", type=str, default="data/sample_c4.jsonl")
    parser.add_argument("--save_path", type=str, default="output.jsonl")
    parser.add_argument("--dataset_begin", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=10)
    parser.add_argument("--normalizers", type=str, default="")

    args = parser.parse_args()
    if "t5" in args.model_name_or_path:
        args.is_seq2seq_model = True
        args.is_decoder_only_model = False
    else:
        args.is_seq2seq_model = False
        args.is_decoder_only_model = True

    main(args)

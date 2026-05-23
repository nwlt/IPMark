import re
import torch
import copy
import gc
import string

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

PUNCT_TOKEN_RE = re.compile(r'^\s*(?:\n|\.{3}|[.?!，,。？！…]+(?:\s)?)\s*$')
TRANSLATOR = str.maketrans('', '', string.punctuation)


def is_punct_token(token_str: str) -> bool:
    return bool(PUNCT_TOKEN_RE.fullmatch(token_str))


def split_sentences(text):
    if not text:
        return []
    sentences = []
    current = ""
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        current += ch
        if is_punct_token(ch):
            sentences.append(current)
            current = ""
            i += 1
            space_chunk = ""
            while i < length and text[i].isspace() and text[i] != "\n":
                space_chunk += text[i]
                i += 1
            current += space_chunk
            continue
        i += 1

    if current.strip():
        sentences.append(current)

    return [s for s in sentences if s.strip()]


def filter_short_word_sentences(sentences):
    filtered = []
    for sent in sentences:
        if len(sent) > 3:
            filtered.append(sent)
    return filtered


def load_model(args, cache_dir=None):
    args.is_seq2seq_model = any(
        (model_type in args.model_name_or_path) for model_type in ["t5", "T0"]
    )
    args.is_decoder_only_model = any(
        (model_type in args.model_name_or_path)
        for model_type in ["gpt", "opt", "bloom", "Llama"]
    )

    if not args.is_seq2seq_model and not args.is_decoder_only_model:
        args.is_decoder_only_model = True

    model_name = args.model_name_or_path

    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                cache_dir=cache_dir,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = args.gpu if torch.cuda.is_available() else "cpu"
        if not args.load_fp16:
            model = model.to(device)
    else:
        device = "cpu"

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
    return model, tokenizer, device


def reorder_cache_manual(past_key_values, beam_indices):
    if hasattr(past_key_values, "key_cache"):
        new_cache = copy.copy(past_key_values)
        new_cache.key_cache = []
        new_cache.value_cache = []

        for layer_idx in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[layer_idx]
            v = past_key_values.value_cache[layer_idx]

            new_k = k.index_select(0, beam_indices)
            new_v = v.index_select(0, beam_indices)

            new_cache.key_cache.append(new_k)
            new_cache.value_cache.append(new_v)

        return new_cache
    else:
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_layer_past = tuple(
                past_state.index_select(0, beam_indices) for past_state in layer_past
            )
            reordered_past += (reordered_layer_past,)
        return reordered_past

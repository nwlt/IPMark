import json
import torch
import nltk
import random
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
from transformers import (
    pipeline,
    T5Tokenizer,
    T5ForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class BatchAttacker:
    def __init__(self, device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        MY_CACHE_DIR = os.environ.get("HF_HOME", "./cache_models")
        os.makedirs(MY_CACHE_DIR, exist_ok=True)

        self.mask_filler = pipeline(
            "fill-mask",
            model="bert-base-uncased",
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={"cache_dir": MY_CACHE_DIR},
        )

        self.en_zh_name = "Helsinki-NLP/opus-mt-en-zh"
        self.zh_en_name = "Helsinki-NLP/opus-mt-zh-en"

        self.trans_tokenizer_en_zh = MarianTokenizer.from_pretrained(
            self.en_zh_name, cache_dir=MY_CACHE_DIR
        )
        self.trans_model_en_zh = MarianMTModel.from_pretrained(
            self.en_zh_name, cache_dir=MY_CACHE_DIR
        ).to(self.device)

        self.trans_tokenizer_zh_en = MarianTokenizer.from_pretrained(
            self.zh_en_name, cache_dir=MY_CACHE_DIR
        )
        self.trans_model_zh_en = MarianMTModel.from_pretrained(
            self.zh_en_name, cache_dir=MY_CACHE_DIR
        ).to(self.device)

        dipper_name = "kalpeshk2011/dipper-paraphraser-xxl"
        self.dipper_tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-v1_1-xxl", cache_dir=MY_CACHE_DIR
        )
        self.dipper_model = T5ForConditionalGeneration.from_pretrained(
            dipper_name, cache_dir=MY_CACHE_DIR
        ).to(self.device)
        self.dipper_model.eval()

    def attack_copy_paste(self, target_text, interference_text, ratio=0.5):
        target_sents = nltk.sent_tokenize(target_text)
        inter_sents = nltk.sent_tokenize(interference_text)

        if not target_sents:
            return target_text
        if not inter_sents:
            return target_text

        n_target = max(1, int(len(target_sents) * ratio))
        needed_inter = max(1, len(target_sents) - n_target)

        while len(inter_sents) < needed_inter:
            inter_sents += inter_sents

        mixed = target_sents[:n_target] + inter_sents[:needed_inter]
        random.shuffle(mixed)
        return " ".join(mixed)

    def attack_synonym(self, text, ratio):
        words = text.split()
        if len(words) < 5:
            return text

        n_mod = max(1, int(len(words) * ratio))
        indices = random.sample(range(len(words)), n_mod)
        new_words = words.copy()

        for idx in indices:
            original = words[idx]
            if not original.isalpha() or len(original) < 3:
                continue

            masked_str = " ".join(new_words[:idx] + ["[MASK]"] + new_words[idx + 1:])
            if len(masked_str) > 512:
                masked_str = masked_str[:512]

            try:
                preds = self.mask_filler(masked_str, top_k=5)
                if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
                    preds = preds[0]

                for p in preds:
                    token = p['token_str']
                    if token.lower() != original.lower() and token.isalpha():
                        new_words[idx] = token
                        break
            except Exception:
                continue
        return " ".join(new_words)

    def attack_rewrite_dipper(self, text):
        sentences = nltk.sent_tokenize(text)
        rewritten_sentences = []
        lexical_code = 40
        order_code = 20

        for sent in sentences:
            if len(sent.strip()) < 2:
                rewritten_sentences.append(sent)
                continue

            prompt = f"lexical = {lexical_code}, order = {order_code} {sent} </s>"
            inputs = self.dipper_tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.dipper_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=512,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                )
            output_text = self.dipper_tokenizer.decode(outputs[0], skip_special_tokens=True)
            rewritten_sentences.append(output_text)

        return " ".join(rewritten_sentences)

    def attack_translation(self, text):
        sentences = nltk.sent_tokenize(text)
        translated_sentences = []

        for sent in sentences:
            if len(sent.strip()) < 2:
                translated_sentences.append(sent)
                continue

            try:
                inputs = self.trans_tokenizer_en_zh(
                    sent, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                with torch.no_grad():
                    zh_out = self.trans_model_en_zh.generate(**inputs)
                zh_text = self.trans_tokenizer_en_zh.decode(zh_out[0], skip_special_tokens=True)

                inputs_back = self.trans_tokenizer_zh_en(
                    zh_text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                with torch.no_grad():
                    en_out = self.trans_model_zh_en.generate(**inputs_back)
                en_text = self.trans_tokenizer_zh_en.decode(en_out[0], skip_special_tokens=True)

                translated_sentences.append(en_text)
            except Exception:
                translated_sentences.append(sent)

        return " ".join(translated_sentences)


def load_interference_corpus(path_input):
    texts = []
    files = []
    if os.path.isfile(path_input):
        files.append(path_input)
    elif os.path.isdir(path_input):
        files = glob.glob(os.path.join(path_input, "*.txt")) + glob.glob(
            os.path.join(path_input, "*.jsonl")
        )
    else:
        return []

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    for line in f:
                        try:
                            item = json.loads(line)
                            content = item.get("output_with_watermark")
                            if content and isinstance(content, str) and len(content.strip()) > 0:
                                raw_sents = nltk.sent_tokenize(content)
                                valid_sents = [s.strip() for s in raw_sents if len(s.strip()) > 5]
                                texts.extend(valid_sents)
                        except (json.JSONDecodeError, KeyError):
                            continue
                else:
                    content = f.read().strip()
                    if content:
                        raw_sents = nltk.sent_tokenize(content)
                        texts.extend([s for s in raw_sents if len(s.strip()) > 5])
        except OSError:
            pass
    return texts


def process_dataset(input_file, output_file, interference_path):
    data_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_lines.append(json.loads(line))

    interference_corpus = load_interference_corpus(interference_path)

    if not interference_corpus:
        for x in data_lines:
            txt = x.get("output_with_watermark", "")
            if txt:
                interference_corpus.extend(nltk.sent_tokenize(txt))

    attacker = BatchAttacker()
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data_lines):
            original_text = item.get("output_with_watermark", "")
            if not original_text or not original_text.strip():
                item["attacked_versions"] = {}
                f_out.write(json.dumps(item) + "\n")
                continue

            attacks = {}
            try:
                attacks["translation_en_zh_en"] = attacker.attack_translation(original_text)
                if interference_corpus:
                    random_sent = random.choice(interference_corpus)
                    attacks["copy_paste_50"] = attacker.attack_copy_paste(
                        original_text, random_sent, ratio=0.5
                    )
                else:
                    attacks["copy_paste_50"] = original_text

                for ratio in [0.05, 0.10, 0.15]:
                    key = f"synonym_{int(ratio * 100)}"
                    attacks[key] = attacker.attack_synonym(original_text, ratio)

                attacks["rewrite_dipper"] = attacker.attack_rewrite_dipper(original_text)
            except Exception as e:
                attacks["error"] = str(e)

            item["attacked_versions"] = attacks
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMark attack generation")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSONL file with watermarked outputs")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file for attacked results")
    parser.add_argument("--interference_path", type=str, default=None,
                        help="Path to interference corpus for copy-paste attack")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    interference_path = args.interference_path or args.input_file
    process_dataset(args.input_file, args.output_file, interference_path)

# IPMark: Sentence-Level Watermarking for LLM Text

IPMark is a sentence-level watermarking method for LLM-generated text. It embeds watermark signals into each sentence during generation by combining **syntactic structure** and **semantic information** via HMAC-based watermarking, and supports watermark detection on the resulting text.

## Algorithm Overview

During LLM text generation, IPMark computes two HMAC watermark signals for each complete sentence:

- **Syntax watermark**: based on the sentence's dependency parse tree structure
- **Semanteme watermark**: based on hypernym abstractions of the subject, verb, and object

A beam search with a watermark-preference selection strategy guides the model to produce sentences that satisfy watermark constraints while preserving text quality. Detection requires no model access — it only recomputes watermark signals for each sentence in the text.

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm

# Download the NLTK punkt tokenizer
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start

### Generate Watermarked Text

```bash
python main.py \
    --model_name_or_path facebook/opt-1.3b \
    --file_path data/sample_c4.jsonl \
    --save_path output.jsonl \
    --use_gpu \
    --gpu cuda:0 \
    --beam_width 10 \
    --accepted_sentences_target 20 \
    --dataset_begin 0 \
    --dataset_end 10
```

### Detect Watermarks

```bash
python detect.py \
    --input_file output.jsonl \
    --output_file detect_result.jsonl
```

### Robustness Evaluation (Attacks)

```bash
# Apply attacks to watermarked text (synonym substitution, back-translation, Dipper paraphrase, etc.)
python attack.py \
    --input_file output.jsonl \
    --output_file attacked.jsonl

# Detect watermarks after attacks
python attack_detect.py \
    --input_file attacked.jsonl \
    --output_file attacked_detect_result.jsonl

# Evaluate AUC / ROC / F1
python attack_eval.py \
    --pos_jsonl attacked_detect_result.jsonl \
    --neg_jsonl natural_text_detect.jsonl
```

### Perplexity Calculation

```bash
python calc_ppl.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_file output.jsonl
```

## Ablation Study

```bash
# --ablation_mode: full / no_quality / no_syntax / no_quality_and_syntax
python ablation_study/generate_ablation.py \
    --model_name_or_path facebook/opt-1.3b \
    --ablation_mode no_quality \
    --file_path data/sample_c4.jsonl
```

## Baseline Comparison

The baseline scripts depend on [MarkLLM](https://github.com/THU-BPM/MarkLLM). Set the `MARKLLM_ROOT` environment variable first:

```bash
export MARKLLM_ROOT=/path/to/MarkLLM

python baseline/baseline_kgw.py   # KGW watermark baseline
python baseline/baseline_no_wm.py # No-watermark baseline
```

## Dataset

`data/sample_c4.jsonl` contains 20 sample lines from the C4 dataset. For the full C4 dataset, see [HuggingFace Datasets](https://huggingface.co/datasets/allenai/c4) or [commoncrawl.org](https://commoncrawl.org/).

## License

MIT License. See [LICENSE](LICENSE).

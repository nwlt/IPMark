import json
import argparse
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(fpr, tpr, thresholds, target_fpr):
    mask = fpr <= target_fpr
    if not np.any(mask):
        return float("nan"), float("nan")
    idx = np.argmax(tpr[mask])
    true_indices = np.where(mask)[0]
    best_i = true_indices[idx]
    return float(tpr[best_i]), float(thresholds[best_i])


def best_f1_from_roc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))

    best = {
        "f1": -1.0,
        "thr": None,
        "precision": None,
        "recall": None,
        "tp": None, "fp": None, "fn": None, "tn": None,
    }

    for i in range(len(thresholds)):
        tp = int(np.rint(tpr[i] * P))
        fp = int(np.rint(fpr[i] * N))

        tp = max(0, min(P, tp))
        fp = max(0, min(N, fp))

        fn = P - tp
        tn = N - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if (f1 > best["f1"]) or (np.isclose(f1, best["f1"]) and recall > (best["recall"] or -1)):
            best.update({
                "f1": float(f1),
                "thr": float(thresholds[i]),
                "precision": float(precision),
                "recall": float(recall),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })

    return best, (fpr, tpr, thresholds)


def transform_direction(scores, score_direction):
    scores = np.asarray(scores, dtype=float)
    if score_direction == "higher_is_more_positive":
        return scores
    elif score_direction == "lower_is_more_positive":
        eps = 1e-300
        return -np.log10(np.clip(scores, eps, None))
    else:
        raise ValueError(
            "score_direction must be 'higher_is_more_positive' or 'lower_is_more_positive'"
        )


def evaluate_binary(pos_scores, neg_scores, title=""):
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)

    y_true = np.r_[np.ones(len(pos_scores), dtype=int), np.zeros(len(neg_scores), dtype=int)]
    y_score = np.r_[pos_scores, neg_scores]

    auc = roc_auc_score(y_true, y_score)
    best_f1_info, (fpr, tpr, thresholds) = best_f1_from_roc(y_true, y_score)

    targets = [0.01, 0.05, 0.10]
    tpr_results = {tfpr: tpr_at_fpr(fpr, tpr, thresholds, tfpr) for tfpr in targets}

    print(f"\n=== {title} ===")
    print(f"Pos count: {len(pos_scores)}  Neg count: {len(neg_scores)}")
    print(f"ROC-AUC: {auc:.6f}")
    for tfpr in targets:
        best_tpr, best_thr = tpr_results[tfpr]
        print(f"TPR@FPR<= {int(tfpr * 100)}% : {best_tpr:.6f}  (Threshold score >= {best_thr:.6f})")

    print("\n--- Best F1 (threshold sweep) ---")
    print(f"Best F1      : {best_f1_info['f1']:.6f}")
    print(f"Best thr     : score >= {best_f1_info['thr']:.6f} -> predict positive")
    print(f"Precision    : {best_f1_info['precision']:.6f}")
    print(f"Recall (TPR) : {best_f1_info['recall']:.6f}")
    print(f"Confusion    : TP={best_f1_info['tp']}, FP={best_f1_info['fp']}, "
          f"FN={best_f1_info['fn']}, TN={best_f1_info['tn']}")

    return {
        "auc": float(auc),
        "best_f1": best_f1_info,
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
    }


def load_pos_scores_by_attack(path, score_key="z_total_rate"):
    out = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"JSON failed: {path} line {lineno}")

            test_idx = obj.get("test_index", lineno)
            for attack_name, inner in obj["results"].items():
                if not isinstance(inner, dict):
                    continue
                if score_key not in inner:
                    continue
                val = inner.get(score_key, None)
                if val is None:
                    continue
                try:
                    out[attack_name].append((test_idx, float(val)))
                except (TypeError, ValueError):
                    continue
    return out


def load_neg_scores_simple(path, score_key="z_total_rate"):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"JSON failed: {path} line {lineno}")

            test_idx = obj.get("test_index", lineno)
            val = obj.get(score_key, None)
            if val is None:
                continue
            try:
                pairs.append((test_idx, float(val)))
            except (TypeError, ValueError):
                continue
    return pairs


def align_by_test_index(pos_pairs, neg_pairs):
    pos_map = {i: s for i, s in pos_pairs}
    neg_map = {i: s for i, s in neg_pairs}
    common = sorted(set(pos_map.keys()) & set(neg_map.keys()))
    pos_scores = [pos_map[i] for i in common]
    neg_scores = [neg_map[i] for i in common]
    return pos_scores, neg_scores, len(common)


def main(pos_jsonl, neg_jsonl, score_key="z_total_rate",
         score_direction="higher_is_more_positive"):
    pos_by_attack = load_pos_scores_by_attack(pos_jsonl, score_key=score_key)
    neg_pairs = load_neg_scores_simple(neg_jsonl, score_key=score_key)

    attacks = sorted(pos_by_attack.keys())
    if "Original" in attacks:
        attacks = ["Original"] + [a for a in attacks if a != "Original"]

    if "Original" in pos_by_attack:
        pos_scores, neg_scores, n_common = align_by_test_index(
            pos_by_attack["Original"], neg_pairs
        )
        pos_scores = transform_direction(pos_scores, score_direction)
        neg_scores = transform_direction(neg_scores, score_direction)
        evaluate_binary(pos_scores, neg_scores, title="Before Attack (Original vs Neg)")

    post_attacks = [a for a in attacks if a != "Original"]
    for attack in post_attacks:
        pos_scores, neg_scores, n_common = align_by_test_index(
            pos_by_attack[attack], neg_pairs
        )
        if n_common == 0:
            continue
        pos_scores = transform_direction(pos_scores, score_direction)
        neg_scores = transform_direction(neg_scores, score_direction)
        evaluate_binary(pos_scores, neg_scores, title=f"After Attack ({attack} vs Neg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMark attack evaluation (AUC/ROC/F1)")
    parser.add_argument("--pos_jsonl", type=str, required=True,
                        help="Detection results on attacked watermarked texts")
    parser.add_argument("--neg_jsonl", type=str, required=True,
                        help="Detection results on natural (unwatermarked) texts")
    parser.add_argument("--score_key", type=str, default="z_total_rate")
    parser.add_argument("--score_direction", type=str, default="higher_is_more_positive")
    args = parser.parse_args()

    if not os.path.exists(args.pos_jsonl):
        raise FileNotFoundError(f"Positive file not found: {args.pos_jsonl}")
    if not os.path.exists(args.neg_jsonl):
        raise FileNotFoundError(f"Negative file not found: {args.neg_jsonl}")

    main(args.pos_jsonl, args.neg_jsonl, args.score_key, args.score_direction)

import argparse
import itertools
import os
from typing import List, Tuple

import pandas as pd
import torch
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.misc import stack_variable_length_tensors

from MafExtractor.maf.models.maf_extractor import MafExtractor
from MafExtractor.maf.utils.aa import seq_to_tensor
from MafExtractor.predict_model.model.MafAFP.MafAFPmodel import MafAFPClassifier


def build_mutants(base_seq: str, target: str = "K", choices: Tuple[str, ...] = ("K", "R", "W")) -> List[str]:
    positions = [i for i, aa in enumerate(base_seq) if aa == target]
    if not positions:
        return []

    mutants = []
    for replacement in itertools.product(choices, repeat=len(positions)):
        seq_list = list(base_seq)
        for pos, aa in zip(positions, replacement):
            seq_list[pos] = aa
        mutants.append("".join(seq_list))
    return mutants


def encode_sequences(sequences: List[str], max_length: int, device: torch.device) -> torch.Tensor:
    tokenizer = get_esmc_model_tokenizers()
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Tokenizer pad token id is None.")

    token_tensors = []
    for seq in sequences:
        seq = seq.replace("_", tokenizer.mask_token)
        ids = tokenizer.encode(
            seq,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        token_tensors.append(torch.tensor(ids, dtype=torch.int64))
    return stack_variable_length_tensors(token_tensors, constant_value=pad).to(device)


def build_maf_batch(sequences: List[str], device: torch.device) -> torch.Tensor:
    xs = [seq_to_tensor(seq) for seq in sequences]
    max_len = max(x.size(0) for x in xs)
    feat_dim = xs[0].size(1)
    batch = torch.zeros(len(xs), max_len, feat_dim, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        batch[i, : x.size(0)] = x
    return batch.to(device)


def main():
    parser = argparse.ArgumentParser(description="Generate K->(R/W) mutants and predict AFP scores.")
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    default_out = os.path.join(root_path, "MafExtractor", "predict_model", "save_model", "mutation_predictions.csv")

    parser.add_argument("--base_seq", type=str, default="VKVKVKYNGTKVKVKV")
    parser.add_argument("--maf_ckpt", type=str, required=True, help="Path to MafExtractor checkpoint (best_maf.pt).")
    parser.add_argument("--clf_ckpt", type=str, required=True, help="Path to classifier checkpoint (e.g. best_AUC.pkl).")
    parser.add_argument("--out_csv", type=str, default=default_out)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--include_reference", action="store_true", help="Include original sequence in output.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    requested_device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; switched to CPU.")

    k_count = args.base_seq.count("K")
    expected_total = 3 ** k_count
    mutants = build_mutants(args.base_seq, target="K", choices=("K", "R", "W"))
    if args.include_reference and args.base_seq not in mutants:
        sequences = [args.base_seq] + mutants
    else:
        sequences = mutants

    if len(sequences) == 0:
        raise ValueError("No mutant sequence was generated. Check the base sequence and target residue.")

    print(f"Base sequence: {args.base_seq}")
    print(f"K positions: {k_count}")
    print(f"Expected combinations (3^K): {expected_total}")
    print(f"Generated combinations: {len(mutants)}")

    maf = MafExtractor().to(requested_device)
    maf.load_state_dict(torch.load(args.maf_ckpt, map_location=requested_device))
    maf.eval()
    for p in maf.parameters():
        p.requires_grad = False

    model_args = {
        "model_name": "esmc_600m",
        "classifier_hidden_ratio": 1,
        "id2label": {"pos": 1, "neg": 0},
        "use_flash_attention": True,
    }
    model = MafAFPClassifier.from_esm_pretrained(**model_args).to(requested_device)
    state_dict = torch.load(args.clf_ckpt, map_location=requested_device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    records = []
    with torch.no_grad():
        for i in range(0, len(sequences), args.batch_size):
            batch_seqs = sequences[i : i + args.batch_size]
            maf_x = build_maf_batch(batch_seqs, requested_device)
            maf_out = maf(maf_x, batch_seqs)
            maf_feat = maf_out["global_feat"]

            seq_tokens = encode_sequences(batch_seqs, args.max_length, requested_device)
            h_seq = model.forward_backbone_only(sequence_tokens=seq_tokens)
            logits = model.forward_classifier_only(h_seq, maf_feat)
            probs = torch.sigmoid(logits).squeeze(-1).detach().cpu().tolist()

            for seq, p in zip(batch_seqs, probs):
                changes = ";".join(
                    f"{idx + 1}:K->{aa}" for idx, aa in enumerate(seq) if args.base_seq[idx] == "K" and aa != "K"
                )
                records.append(
                    {
                        "sequence": seq,
                        "probability": float(p),
                        "pred_label@0.5": int(p >= 0.5),
                        "changes": changes,
                    }
                )

    df = pd.DataFrame(records).sort_values("probability", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()

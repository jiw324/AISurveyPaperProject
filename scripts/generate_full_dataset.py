"""
Unified data generator for prompt injection detection.

Steps:
1) Generate context-rich realistic data
2) Paraphrase/perturb to make it harder
3) Produce binary (train/val/test) and 6-class variants

Outputs:
- data/train.jsonl, data/val.jsonl, data/test.jsonl           (hard binary)
- data/train_multiclass.jsonl, val_multiclass.jsonl, test_multiclass.jsonl
- Intermediate: *_realistic.jsonl and *_hard.jsonl
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List

from generate_realistic_data import (
    generate_legitimate_examples,
    generate_injection_examples,
    save_dataset,
)
from generate_paraphrased_data import make_hard_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ATTACK_LABELS = {
    "none": 0,
    "goal_hijacking": 1,
    "context_manipulation": 2,
    "jailbreaking": 3,
    "multi_turn": 4,
    "obfuscated": 5,
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_realistic_split(split: str, total_size: int, output_dir: Path):
    """Generate realistic split and save to *_realistic.jsonl."""
    n_legit = total_size // 2
    n_inject = total_size - n_legit

    logger.info(f"[{split}] Generating realistic data: {total_size} (legit={n_legit}, inj={n_inject})")
    legit_examples = generate_legitimate_examples(n_legit)
    inject_examples = generate_injection_examples(n_inject)

    all_examples: List[Dict] = legit_examples + inject_examples
    random.shuffle(all_examples)

    out_path = output_dir / f"{split}_realistic.jsonl"
    save_dataset(all_examples, out_path)
    logger.info(f"[{split}] ✓ Saved {len(all_examples)} -> {out_path}")
    return out_path


def copy_file(src: Path, dst: Path):
    shutil.copyfile(src, dst)
    logger.info(f"Copied {src.name} -> {dst}")


def convert_to_multiclass(input_path: Path, output_path: Path):
    """Convert binary examples to 6-way labels based on attack_type."""
    allowed = set(ATTACK_LABELS.keys())
    count = 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            ex = json.loads(line)
            label = ex.get("label", "legitimate")
            attack_type = ex.get("attack_type", "none")

            if label == "legitimate":
                label = "none"
                attack_type = "none"
            else:
                if attack_type not in allowed or attack_type == "none":
                    attack_type = "goal_hijacking"
                label = attack_type

            ex["label"] = label
            ex["attack_type"] = attack_type
            fout.write(json.dumps(ex) + "\n")
            count += 1
    logger.info(f"Multiclass: {input_path.name} -> {output_path} ({count} rows)")


def parse_args():
    p = argparse.ArgumentParser(description="Generate full dataset (realistic + hard + multiclass)")
    p.add_argument("--train-size", type=int, default=50000, help="Train split size")
    p.add_argument("--val-size", type=int, default=10000, help="Val split size")
    p.add_argument("--test-size", type=int, default=10000, help="Test split size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    data_dir = Path("data")
    ensure_dir(data_dir)

    sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }

    logger.info("=" * 80)
    logger.info("STEP 1: Generate realistic data")
    logger.info("=" * 80)
    realistic_paths = {
        split: generate_realistic_split(split, size, data_dir)
        for split, size in sizes.items()
    }

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Paraphrase to HARD data")
    logger.info("=" * 80)
    hard_paths = {}
    for split, src in realistic_paths.items():
        dst = data_dir / f"{split}_hard.jsonl"
        make_hard_split(src, dst)
        hard_paths[split] = dst

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Set canonical binary files (hard)")
    logger.info("=" * 80)
    copy_file(hard_paths["train"], data_dir / "train.jsonl")
    copy_file(hard_paths["val"], data_dir / "val.jsonl")
    copy_file(hard_paths["test"], data_dir / "test.jsonl")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Convert to multiclass (6 labels)")
    logger.info("=" * 80)
    convert_to_multiclass(hard_paths["train"], data_dir / "train_multiclass.jsonl")
    convert_to_multiclass(hard_paths["val"], data_dir / "val_multiclass.jsonl")
    convert_to_multiclass(hard_paths["test"], data_dir / "test_multiclass.jsonl")

    logger.info("\nAll done.")
    logger.info("Binary (hard): data/train.jsonl, data/val.jsonl, data/test.jsonl")
    logger.info("Multiclass:    data/train_multiclass.jsonl, val_multiclass.jsonl, test_multiclass.jsonl")
    logger.info("Intermediate:  *_realistic.jsonl, *_hard.jsonl")


if __name__ == "__main__":
    main()



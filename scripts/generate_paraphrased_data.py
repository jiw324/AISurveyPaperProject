"""Generate a harder, paraphrased dataset to defeat keyword memorization.

This builds on top of the existing realistic generator by:
- Paraphrasing with synonym substitutions
- Injecting mild typos / casing changes / filler text
- Adding overlap between legitimate and attack phrasing

Outputs:
  data/train_hard.jsonl
  data/val_hard.jsonl
  data/test_hard.jsonl
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

random.seed(1234)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small synonym bank for lightweight paraphrasing (no external APIs)
# ---------------------------------------------------------------------------
SYNONYMS = {
    "ignore": ["disregard", "skip", "set aside", "overlook"],
    "instructions": ["directions", "guidelines", "prompts", "rules"],
    "reveal": ["show", "expose", "share", "disclose"],
    "system": ["setup", "platform", "stack", "architecture"],
    "prompt": ["message", "input", "query"],
    "hack": ["exploit", "breach", "break into"],
    "admin": ["administrator", "superuser"],
    "secret": ["confidential info", "hidden details"],
    "explain": ["describe", "clarify", "outline", "break down"],
    "help": ["assist", "support", "guide"],
    "question": ["query", "ask", "inquiry"],
}


def random_synonym(word: str) -> str:
    if word.lower() in SYNONYMS:
        return random.choice(SYNONYMS[word.lower()])
    return word


def perturb_text(text: str) -> str:
    """Apply lightweight perturbations to make patterns less consistent."""
    words = text.split()
    # random synonym swaps
    for i in range(len(words)):
        if random.random() < 0.15:
            words[i] = random_synonym(words[i])
    text = " ".join(words)

    # random casing changes
    if random.random() < 0.2:
        text = text.capitalize()
    if random.random() < 0.1:
        text = text.upper()

    # inject minor typos / filler
    if random.random() < 0.15:
        fillers = [
            "uh", "btw", "tbh", "kind of", "like", "sort of",
            "pls", "thx", "lol", "idk"
        ]
        insert_pos = random.randint(0, len(text))
        text = text[:insert_pos] + " " + random.choice(fillers) + " " + text[insert_pos:]

    return text.strip()


def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with path.open() as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: Path):
    with path.open("w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def add_overlap_legit_attack(examples: List[Dict]) -> List[Dict]:
    """Make some legitimate prompts contain 'ignore/forget' style language."""
    result = []
    overlap_phrases = [
        "Ignore any typos, but",
        "Forget my last typo and",
        "Disregard any mistakes and",
        "Set aside the previous phrasing and",
    ]
    for ex in examples:
        if ex["label"] == "legitimate" and random.random() < 0.25:
            ex = dict(ex)
            prefix = random.choice(overlap_phrases)
            ex["text"] = f"{prefix} {ex['text']}"
            ex["metadata"]["overlap"] = True
        result.append(ex)
    return result


def make_hard_split(src_path: Path, out_path: Path):
    data = load_jsonl(src_path)
    hard = []
    for ex in tqdm(data, desc=f"Paraphrasing {src_path.name}"):
        ex = dict(ex)
        ex["text"] = perturb_text(ex["text"])
        ex.setdefault("metadata", {})
        ex["metadata"]["perturbed"] = True
        hard.append(ex)

    # Add overlap to increase ambiguity
    hard = add_overlap_legit_attack(hard)

    random.shuffle(hard)
    save_jsonl(hard, out_path)
    logger.info(f"Saved {len(hard)} examples to {out_path}")


def main():
    base_dir = Path("data")
    inputs = {
        "train": base_dir / "train_realistic.jsonl",
        "val": base_dir / "val_realistic.jsonl",
        "test": base_dir / "test_realistic.jsonl",
    }
    outputs = {
        "train": base_dir / "train_hard.jsonl",
        "val": base_dir / "val_hard.jsonl",
        "test": base_dir / "test_hard.jsonl",
    }

    # Check existence
    for split, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/generate_realistic_data.py first."
            )

    logger.info("Generating HARD paraphrased dataset from realistic data...")
    for split in ["train", "val", "test"]:
        make_hard_split(inputs[split], outputs[split])

    logger.info("\nDone. To use the harder data, point run_experiment.py to:")
    logger.info("  data/train_hard.jsonl, data/val_hard.jsonl, data/test_hard.jsonl")


if __name__ == "__main__":
    main()


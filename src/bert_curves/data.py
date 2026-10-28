from pathlib import Path
import numpy as np
import pandas as pd

DATASETS = {"imdb": ("stanfordnlp/imdb", "plain_text"),
            "ag_news": ("fancyzhx/ag_news", "data")}


def load_data(dataset, cache_dir="data"):
    if dataset not in DATASETS:
        raise ValueError(f"Dataset deve ser um de {list(DATASETS)}")
    repo, subset = DATASETS[dataset]
    frames = []
    for split in ("train", "test"):
        path = Path(cache_dir) / f"{dataset}_{split}.parquet"
        if not path.exists():
            url = (f"https://huggingface.co/datasets/{repo}/resolve/main/"
                   f"{subset}/{split}-00000-of-00001.parquet")
            frame = pd.read_parquet(url)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame[["text", "label"]].to_parquet(path, index=False)
        frames.append(pd.read_parquet(path).reset_index(drop=True))
    return tuple(frames)


def balanced_order(labels, seed):
    """Prefixos aninhados e estratificados; preserva classes desiguais."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(np.flatnonzero(labels == c)))
              for c in np.unique(labels)]
    order = []
    for i in range(max(map(len, groups))):
        for c in rng.permutation(len(groups)):
            if i < len(groups[c]):
                order.append(groups[c][i])
    return np.asarray(order, dtype=int)


def select_test(frame, size, seed):
    if size is None:
        return frame.copy()
    if size < frame.label.nunique() or size > len(frame):
        raise ValueError("test_size deve cobrir as classes e caber no split oficial")
    return frame.iloc[balanced_order(frame.label, seed)[:size]].reset_index(drop=True)

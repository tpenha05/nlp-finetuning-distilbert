import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from transformers import DistilBertConfig, DistilBertTokenizerFast
from .data import balanced_order, load_data, select_test
from .models import fit, make_classifier, predict, seed_all, tensor_data

REGIMES = ("bow", "scratch", "frozen", "finetuned")


def validate_config(cfg, train, test):
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Informe sementes distintas")
    if not cfg["sizes"] or sorted(set(cfg["sizes"])) != cfg["sizes"]:
        raise ValueError("sizes deve ser crescente e sem repetições")
    if min(cfg["sizes"]) < train.label.nunique() or max(cfg["sizes"]) > len(train):
        raise ValueError("Tamanhos devem cobrir as classes e caber no treino")
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or not 2 <= cfg["max_length"] <= 512:
        raise ValueError("epochs/batch_size positivos e max_length entre 2 e 512")
    classes = sorted(train.label.unique().tolist())
    if classes != list(range(len(classes))) or not set(test.label).issubset(classes):
        raise ValueError("Rótulos devem ser contíguos a partir de zero")


def digest_frame(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def run(cfg, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.csv").exists():
        raise FileExistsError(f"{out}/metrics.csv já existe; use outra pasta")
    torch.set_num_threads(cfg["threads"])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"]
    train, official_test = load_data(cfg["dataset"])
    test = select_test(official_test, cfg["test_size"], cfg["test_seed"])
    validate_config(cfg, train, test)
    model_cfg = DistilBertConfig.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["model_id"], revision=model_cfg._commit_hash)
    orders = {seed: balanced_order(train.label, seed)[:max(cfg["sizes"])] for seed in cfg["seeds"]}
    test_data = tensor_data(tokenizer, test.text, test.label, cfg["max_length"])
    manifest = {
        "config": cfg, "device": device, "python": platform.python_version(),
        "model_commit": model_cfg._commit_hash,
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "numpy", "pandas", "scikit-learn")},
        "train_sha256": digest_frame(train), "test_sha256": digest_frame(test),
        "official_train_size": len(train), "official_test_size": len(official_test),
        "evaluated_test_size": len(test), "train_order_indices": {str(k):v.tolist() for k,v in orders.items()},
        "test_indices": (balanced_order(official_test.label, cfg["test_seed"])[:cfg["test_size"]].tolist()
                         if cfg["test_size"] is not None else list(range(len(official_test)))),
        "status": "running",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for seed in cfg["seeds"]:
        for n in cfg["sizes"]:
            subset = train.iloc[orders[seed][:n]]
            train_data = tensor_data(tokenizer, subset.text, subset.label, cfg["max_length"])
            for regime in REGIMES:
                seed_all(seed)
                print(f"dataset={cfg['dataset']} seed={seed} n={n} model={regime}", flush=True)
                started = time.perf_counter()
                losses = []
                if regime == "bow":
                    model = make_pipeline(CountVectorizer(ngram_range=(1, 2), max_features=30000),
                                          LogisticRegression(max_iter=2000, random_state=seed))
                    model.fit(subset.text, subset.label)
                    train_pred, test_pred = model.predict(subset.text), model.predict(test.text)
                    parameters = None
                else:
                    model = make_classifier(model_cfg, train.label.nunique(), regime, cfg["model_id"])
                    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    losses = fit(model, train_data, cfg, regime, device, seed)
                    train_pred = predict(model, train_data, cfg["batch_size"], device)
                    test_pred = predict(model, test_data, cfg["batch_size"], device)
                row = {"dataset": cfg["dataset"], "seed": seed, "train_size": n, "model": regime,
                       "test_size": len(test), "train_accuracy": accuracy_score(subset.label, train_pred),
                       "accuracy": accuracy_score(test.label, test_pred),
                       "f1_macro": f1_score(test.label, test_pred, average="macro", zero_division=0),
                       "seconds": time.perf_counter() - started, "trainable_parameters": parameters,
                       "epoch_losses": json.dumps(losses)}
                rows.append(row)
                pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
                print(f"  accuracy={row['accuracy']:.3f} f1={row['f1_macro']:.3f} seconds={row['seconds']:.1f}", flush=True)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    manifest["status"] = "complete"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    from .report import generate_report
    generate_report(out)


def main():
    parser = argparse.ArgumentParser(description="Execute as quatro curvas de aprendizado")
    parser.add_argument("--config", default="configs/quick.json")
    parser.add_argument("--output", default="results/custom/quick")
    parser.add_argument("--dataset", choices=["imdb", "ag_news"])
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.dataset:
        cfg["dataset"] = args.dataset
    run(cfg, args.output)


if __name__ == "__main__":
    main()

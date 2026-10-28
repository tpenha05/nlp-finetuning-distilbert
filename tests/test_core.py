import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset
from transformers import DistilBertConfig, DistilBertModel
from bert_curves.data import balanced_order, select_test
from bert_curves.models import BertClassifier, fit, seed_all
from bert_curves.experiment import validate_config
from bert_curves.report import aggregate, validate_results


def test_nested_stratified_sampling_keeps_every_index():
    labels = np.array([0]*10 + [1]*10 + [2]*5)
    order = balanced_order(labels, 42)
    assert sorted(order) == list(range(25))
    assert np.array_equal(order, balanced_order(labels, 42))
    assert not np.array_equal(order, balanced_order(labels, 43))
    for n in (3, 6, 9, 12, 15):
        counts = np.bincount(labels[order[:n]], minlength=3)
        assert counts.max() - counts.min() <= 1
    assert set(order[:6]).issubset(order[:12])


def test_test_selection_is_fixed_and_balanced():
    df = pd.DataFrame({"text": [str(i) for i in range(20)], "label": [0]*10+[1]*10})
    selected = select_test(df, 6, 2026)
    pd.testing.assert_frame_equal(selected, select_test(df, 6, 2026))
    assert selected.label.value_counts().to_dict() == {0:3, 1:3}
    with pytest.raises(ValueError):
        select_test(df, 30, 2026)


def tiny_encoder():
    return DistilBertModel(DistilBertConfig(vocab_size=30, dim=16, hidden_dim=32,
                           n_layers=1, n_heads=2, max_position_embeddings=16))


@pytest.mark.parametrize("frozen", [True, False])
def test_gradient_and_weight_updates_obey_regime(frozen):
    torch.set_num_threads(1)
    seed_all(42)
    model = BertClassifier(tiny_encoder(), 2, frozen=frozen)
    before = {k:v.clone() for k,v in model.encoder.state_dict().items()}
    head_before = model.head.weight.clone()
    ids = torch.randint(0, 30, (8, 8))
    ds = TensorDataset(ids, torch.ones_like(ids), torch.tensor([0,1]*4))
    cfg = dict(batch_size=4, epochs=1, learning_rate=.001,
               scratch_learning_rate=.001, head_learning_rate=.01, weight_decay=.01)
    fit(model, ds, cfg, "frozen" if frozen else "scratch", "cpu", 42)
    changed = any(not torch.equal(v, before[k]) for k,v in model.encoder.state_dict().items())
    assert changed is (not frozen)
    assert not torch.equal(head_before, model.head.weight)
    model.train()
    assert model.encoder.training is (not frozen)


def test_report_rejects_incomplete_runs_and_aggregates_real_seeds():
    cfg = {"sizes": [50], "seeds": [42,43]}
    rows = [dict(model=m, train_size=50, seed=s, accuracy=.5 + .1*(s==43), f1_macro=.5)
            for m in ("bow","scratch","frozen","finetuned") for s in cfg["seeds"]]
    df = pd.DataFrame(rows)
    manifest = {"config":cfg, "status":"complete"}
    validate_results(df, manifest)
    summary = aggregate(df)
    assert np.allclose(summary.accuracy_mean, .55)
    assert np.allclose(summary.accuracy_std, np.sqrt(.005))
    with pytest.raises(ValueError):
        validate_results(df.iloc[:-1], manifest)
    with pytest.raises(ValueError):
        validate_results(pd.concat([df,df.iloc[:1]]), manifest)


def test_config_rejects_oversized_subsample():
    cfg = json.loads(Path("configs/quick.json").read_text())
    train = pd.DataFrame({"label":[0,1]*10})
    with pytest.raises(ValueError):
        validate_config(cfg, train, train)


def test_pretrained_factory_receives_explicit_model_id(monkeypatch):
    from bert_curves.models import make_classifier
    calls = []
    def fake_load(model_id, **kwargs):
        calls.append(model_id)
        return tiny_encoder()
    monkeypatch.setattr(DistilBertModel, "from_pretrained", fake_load)
    cfg = tiny_encoder().config
    cfg._commit_hash = "test-revision"
    model = make_classifier(cfg, 2, "frozen", "distilbert/distilbert-base-uncased")
    assert calls == ["distilbert/distilbert-base-uncased"]
    assert not any(p.requires_grad for p in model.encoder.parameters())

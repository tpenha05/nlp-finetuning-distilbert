import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertModel


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BertClassifier(nn.Module):
    """Mesmo encoder e cabeça linear nos três regimes; usa o token CLS."""
    def __init__(self, encoder, classes, frozen=False):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.dim, classes)
        self.frozen = frozen
        if frozen:
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, ids, masks):
        if self.frozen:
            with torch.no_grad():
                features = self.encoder(ids, attention_mask=masks).last_hidden_state[:, 0]
        else:
            features = self.encoder(ids, attention_mask=masks).last_hidden_state[:, 0]
        return self.head(features)


def make_classifier(config, classes, regime, model_id):
    if regime == "scratch":
        encoder = DistilBertModel(config)
    else:
        encoder = DistilBertModel.from_pretrained(model_id, revision=config._commit_hash)
    return BertClassifier(encoder, classes, frozen=regime == "frozen")


def tensor_data(tokenizer, texts, labels, max_length):
    encoded = tokenizer(list(texts), padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"],
                         torch.tensor(np.asarray(labels), dtype=torch.long))


def fit(model, dataset, cfg, regime, device, seed):
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    lr = cfg["scratch_learning_rate"] if regime == "scratch" else cfg["learning_rate"]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": lr},
        {"params": model.head.parameters(), "lr": cfg["head_learning_rate"]},
    ], weight_decay=cfg["weight_decay"])
    model.to(device)
    losses = []
    for _ in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for ids, masks, labels in loader:
            ids, masks, labels = ids.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(ids, masks), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
        losses.append(total / len(dataset))
    return losses


@torch.inference_mode()
def predict(model, dataset, batch_size, device):
    model.eval()
    predictions = []
    for ids, masks, _ in DataLoader(dataset, batch_size=batch_size):
        predictions.extend(model(ids.to(device), masks.to(device)).argmax(-1).cpu().tolist())
    return np.asarray(predictions)

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parent


def binary_roc_auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    positives = labels.sum()
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, labels.size + 1)
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


class Dataset:
    def __init__(self, name: str, device: torch.device):
        if name == 'synthetic':
            generator = np.random.default_rng(0)
            nodes, features, classes = 32, 8, 3
            data = {
                'node_features': generator.normal(size=(nodes, features)).astype(np.float32),
                'node_labels': (np.arange(nodes) % classes).astype(np.int64),
                'edges': np.column_stack([np.arange(nodes), np.roll(np.arange(nodes), -1)]),
                'train_masks': np.array([[True] * 20 + [False] * 12]),
                'val_masks': np.array([[False] * 20 + [True] * 6 + [False] * 6]),
                'test_masks': np.array([[False] * 26 + [True] * 6]),
            }
        else:
            path = PROJECT_ROOT / 'data' / f'{name.replace("-", "_")}.npz'
            if not path.exists():
                raise FileNotFoundError(f'Dataset file not found: {path}.')
            data = np.load(path)

        features = torch.as_tensor(data['node_features'], dtype=torch.float32)
        labels = torch.as_tensor(data['node_labels'])
        edges = torch.as_tensor(data['edges'], dtype=torch.long)
        if 'directed' not in name:
            edges = torch.unique(torch.cat([edges, edges[:, [1, 0]]]), dim=0)

        self.name = name
        self.edge_index = edges.T.contiguous().to(device)
        self.node_features = features.to(device)
        self.labels = labels.to(device)
        self.train_idx_list = [torch.where(torch.as_tensor(mask))[0].to(device) for mask in data['train_masks']]
        self.val_idx_list = [torch.where(torch.as_tensor(mask))[0].to(device) for mask in data['val_masks']]
        self.test_idx_list = [torch.where(torch.as_tensor(mask))[0].to(device) for mask in data['test_masks']]
        self.num_data_splits, self.cur_data_split = len(self.train_idx_list), 0
        self.num_node_features = self.node_features.shape[1]
        self.num_targets = 1 if labels.unique().numel() == 2 else int(labels.unique().numel())
        if self.num_targets == 1:
            self.labels, self.loss_fn, self.metric = self.labels.float(), F.binary_cross_entropy_with_logits, 'roc_auc'
        else:
            self.labels, self.loss_fn, self.metric = self.labels.long(), F.cross_entropy, 'accuracy'

    @property
    def train_idx(self): return self.train_idx_list[self.cur_data_split]
    @property
    def val_idx(self): return self.val_idx_list[self.cur_data_split]
    @property
    def test_idx(self): return self.test_idx_list[self.cur_data_split]

    def loss(self, logits, index):
        return self.loss_fn(logits[index], self.labels[index])

    def compute_metrics(self, logits):
        if self.num_targets == 1:
            scores, labels = logits.detach().cpu().numpy(), self.labels.detach().cpu().numpy()
            metric = lambda index: binary_roc_auc(labels[index.cpu().numpy()], scores[index.cpu().numpy()])
        else:
            predictions = logits.argmax(dim=-1)
            metric = lambda index: (predictions[index] == self.labels[index]).float().mean().item()
        return {f'train_{self.metric}': metric(self.train_idx), f'val_{self.metric}': metric(self.val_idx), f'test_{self.metric}': metric(self.test_idx)}

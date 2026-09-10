import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from datasets import Dataset
from model import GSFGNN


def parse_args():
    parser = argparse.ArgumentParser(description='Train GSF-GNN on a local NPZ node-classification dataset.')
    parser.add_argument('--dataset', default='actor', help='Dataset basename in data/, without .npz.')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:<index>.')
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--num-steps', type=int, default=200)
    parser.add_argument('--num-runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments'))
    parser.add_argument('--disable-filters', action='store_true')
    parser.add_argument('--disable-combinations', action='store_true')
    parser.add_argument('--no-global-node', action='store_true')
    return parser.parse_args()


def resolve_device(value):
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_global_node(dataset):
    features = dataset.node_features
    node_id = features.shape[0]
    destination = torch.arange(node_id, device=features.device)
    source = torch.full_like(destination, node_id)
    dataset.edge_index = torch.cat([dataset.edge_index, torch.stack([source, destination])], dim=1)
    dataset.node_features = torch.cat([features, features.max(dim=0).values.unsqueeze(0)], dim=0)


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    logits, _ = model(dataset.edge_index, dataset.node_features)
    metrics = dataset.compute_metrics(logits)
    return metrics, dataset.loss(logits, dataset.train_idx).item()


def run_once(args, run_index, device):
    set_seed(args.seed + run_index)
    dataset = Dataset(args.dataset, device)
    dataset.cur_data_split = run_index % dataset.num_data_splits
    if not args.no_global_node:
        add_global_node(dataset)
    xx_initial = dataset.node_features @ dataset.node_features.T
    model = GSFGNN(
        input_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        output_dim=dataset.num_targets,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        xx_initial=xx_initial,
        use_filters=not args.disable_filters,
        use_combinations=not args.disable_combinations,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {'val': float('-inf'), 'test': float('-inf'), 'step': None}
    for step in trange(1, args.num_steps + 1, desc=f'run {run_index + 1}', leave=False):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(dataset.edge_index, dataset.node_features)
        loss = dataset.loss(logits, dataset.train_idx)
        loss.backward()
        optimizer.step()
        metrics, _ = evaluate(model, dataset)
        val = metrics[f'val_{dataset.metric}']
        if val > best['val']:
            best = {'val': val, 'test': metrics[f'test_{dataset.metric}'], 'step': step}
    return best, dataset.metric


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Using device: {device}')
    outcomes = [run_once(args, run, device) for run in range(args.num_runs)]
    results = [outcome[0] for outcome in outcomes]
    summary = {
        'dataset': args.dataset,
        'metric': outcomes[0][1],
        'runs': results,
        'mean_test': float(np.mean([result['test'] for result in results])),
        'std_test': float(np.std([result['test'] for result in results])),
    }
    output_path = args.output_dir / f'{args.dataset}_results.json'
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Saved results to {output_path}')


if __name__ == '__main__':
    main()

import torch

from model import GSFGNN


def main():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    features = torch.randn(4, 8)
    model = GSFGNN(
        input_dim=8,
        hidden_dim=8,
        output_dim=3,
        num_layers=1,
        num_heads=2,
        xx_initial=features @ features.T,
    )
    logits, _ = model(edge_index, features)
    logits.sum().backward()
    print(f'GSF-GNN smoke test passed; logits shape: {tuple(logits.shape)}')


if __name__ == '__main__':
    main()

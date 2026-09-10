import torch
from torch import nn

from modules import MaxwellDemonFilter, ResidualModule


NORMALIZATION = {
    'none': nn.Identity,
    'layernorm': nn.LayerNorm,
    'batchnorm': nn.BatchNorm1d,
}


class GSFGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads,
                 hidden_dim_multiplier=1.0, normalization='layernorm', dropout=0.4,
                 xx_initial=None, use_filters=True, use_combinations=True):
        super().__init__()
        if xx_initial is None:
            raise ValueError('xx_initial is required for global structure propagation.')
        try:
            norm = NORMALIZATION[normalization.lower()]
        except KeyError as error:
            raise ValueError(f'Unsupported normalization: {normalization}') from error
        num_nodes = xx_initial.shape[0]
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layers = nn.ModuleList([
            ResidualModule(
                MaxwellDemonFilter,
                normalization=norm,
                dim=hidden_dim,
                xx_initial=xx_initial,
                hidden_dim_multiplier=hidden_dim_multiplier,
                num_heads=num_heads,
                num_nodes=num_nodes,
                dropout=dropout,
                use_filters=use_filters,
                use_combinations=use_combinations,
            )
            for _ in range(num_layers)
        ])
        self.output_normalization = norm(hidden_dim * (num_layers + 1))
        self.output_linear = nn.Linear(hidden_dim * (num_layers + 1), output_dim)

    def forward(self, edge_index, x):
        x = self.activation(self.input_dropout(self.input_linear(x)))
        all_layers = [x]
        for layer in self.layers:
            x = layer(edge_index, x)
            all_layers.append(x)
        representation = self.output_normalization(torch.cat(all_layers, dim=-1))
        logits = self.output_linear(representation)
        return logits.squeeze(-1), representation

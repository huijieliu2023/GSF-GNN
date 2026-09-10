import torch
from torch import nn


def _check_dim_and_num_heads_consistency(dim: int, num_heads: int) -> None:
    if dim % num_heads != 0:
        raise ValueError('hidden_dim must be divisible by num_heads.')


def edge_softmax(scores, destination, num_nodes):
    """Softmax over incoming edges for each destination node and attention head."""
    index = destination[:, None].expand_as(scores)
    maximum = torch.full((num_nodes, scores.shape[1]), -torch.inf, device=scores.device, dtype=scores.dtype)
    maximum.scatter_reduce_(0, index, scores, reduce='amax', include_self=True)
    exponentials = torch.exp(scores - maximum[destination])
    denominator = torch.zeros_like(maximum)
    denominator.scatter_add_(0, index, exponentials)
    return exponentials / denominator[destination].clamp_min(torch.finfo(scores.dtype).eps)


def aggregate(destination, values, num_nodes):
    output = values.new_zeros((num_nodes, *values.shape[1:]))
    output.index_add_(0, destination, values)
    return output


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Dropout(dropout), nn.GELU(),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualModule(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, edge_index, x):
        return x + self.module(edge_index, self.normalization(x))


class MaxwellDemonFilter(nn.Module):
    '''Structure-aware propagation and feature-augmented update used by GSF-GNN.'''

    def __init__(self, dim, xx_initial, hidden_dim_multiplier, num_heads, num_nodes,
                 dropout, use_filters, use_combinations):
        super().__init__()
        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.num_nodes = num_nodes
        self.use_filters, self.use_combinations = use_filters, use_combinations
        self.fc = nn.Linear(dim, dim)
        self.attn_linear_u = nn.Linear(dim, num_heads)
        self.attn_linear_v = nn.Linear(dim, num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(0.2)
        self.get_laplace = nn.Linear(num_nodes, num_heads)
        self.chaos_factor = nn.Parameter(torch.zeros(1))
        self.r = nn.Parameter(torch.tensor([0.5]))
        self.filter = nn.Linear(num_heads * 2, 1)
        self.feed_forward = FeedForwardModule(
            dim, hidden_dim_multiplier, dropout, 4 if use_combinations else 2
        )
        self.register_buffer('xx_initial', xx_initial, persistent=False)

    def forward(self, edge_index, x):
        source, destination = edge_index
        x = self.fc(x)
        scores = self.attn_act(self.attn_linear_u(x)[source] + self.attn_linear_v(x)[destination])
        attention = edge_softmax(scores, destination, self.num_nodes)
        if self.use_filters:
            global_state = self.get_laplace(self.xx_initial)
            if self.training:
                global_state = global_state + torch.randn_like(global_state) * self.chaos_factor
            gate_input = torch.cat([global_state[source], global_state[destination]], dim=-1)
            gate = torch.sigmoid(self.filter(gate_input))
            attention = attention * (1 - torch.sigmoid(self.r) + torch.sigmoid(self.r) * gate)

        heads = x.reshape(-1, self.head_dim, self.num_heads)
        message = aggregate(destination, heads[source] * attention[:, None, :], self.num_nodes).reshape(-1, self.dim)
        features = [x, message]
        if self.use_combinations:
            degree_in = torch.bincount(destination, minlength=self.num_nodes).to(x.dtype).clamp_min(1)
            mean = aggregate(destination, heads[source], self.num_nodes) / degree_in[:, None, None]
            degree_out = torch.bincount(source, minlength=self.num_nodes).to(x.dtype).clamp_min(1)
            weights = (degree_out[source] * degree_out[destination]).rsqrt()
            normalized = aggregate(destination, x[source] * weights[:, None], self.num_nodes)
            features.extend([mean.reshape(-1, self.dim), normalized])
        return self.feed_forward(torch.cat(features, dim=-1))

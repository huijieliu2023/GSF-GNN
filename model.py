from torch import nn
import torch
import dgl

from modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, SAGEModule, GATModule, GATSepModule,
                     TransformerAttentionModule, TransformerAttentionSepModule)
from modules import MaxwellDemonFilter

MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    'GAT-sep': [GATSepModule],
    'GT': [TransformerAttentionModule, FeedForwardModule],
    'GT-sep': [TransformerAttentionSepModule, FeedForwardModule]
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}

class GMP_GNN(nn.Module):
    def __init__(self, args, input_dim, hidden_dim,output_dim,hidden_dim_multiplier, num_heads,num_nodes,xx_initial, normalization, dropout,number_of_edges, num_layers):
        super(MyModel, self).__init__()
        self.args = args
        self.xx = xx_initial
        normalization = NORMALIZATION[normalization]
        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            residual_module = GMP_GNN_layer(module=GMP_GNN_layer,
                                                    normalization=normalization,
                                                    xx_initial = xx_initial,
                                                    dim=hidden_dim,
                                                    hidden_dim_multiplier=hidden_dim_multiplier,
                                                    num_heads=num_heads,
                                                    num_nodes = num_nodes,
                                                    dropout=dropout,
                                                    number_of_edges = number_of_edges,
                                                    args= self.args
                                                    )

            self.residual_modules.append(residual_module)

        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.output_normalization = normalization(hidden_dim * (num_layers+1))
        self.output_linear = nn.Linear(in_features=hidden_dim * (num_layers+1), out_features=output_dim)

    def forward(self,graph,x):
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x) 
        x_all =  x
        for residual_module in self.residual_modules:
            x = residual_module(graph, x)
            x_all = torch.cat([x_all,x], dim = -1)

        x_all_0 = self.output_normalization(x_all)
        x_all = self.output_linear(x_all_0).squeeze(1)

        return x_all,x_all_0


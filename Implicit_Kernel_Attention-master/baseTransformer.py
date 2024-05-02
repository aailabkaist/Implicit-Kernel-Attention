import torch
import torch.nn as nn
import math

from MHA import MultiHeadAttentionLayer
from Layer import PositionwiseFeedforwardLayer

class baseTransformer(nn.Module):
    def __init__(self,
                 args,
                 num_data,
                 input_dim,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,):
        super().__init__()

        self.args = args
        self.num_data = num_data
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, 1)
        self.pos_embedding = nn.Embedding(num_data, 1)
        self.pos_embedding.weight.requires_grad = False

        self.w1 = torch.nn.Parameter(torch.empty(hid_dim // n_heads, n_heads, args.M), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.empty(hid_dim // n_heads, n_heads, args.M), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.layers = nn.ModuleList([EncoderLayer(args,
                                                  device,
                                                  hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  self.w1,
                                                  self.w2)
                                     for _ in range(n_layers)])
        self.fc_in = nn.Linear(input_dim, args.KEY_DIM)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.key_dim = args.KEY_DIM

    def forward(self, x):
        # x : [maxlen,feature_dim]
        x = x.unsqueeze(0) # x : [1, src len, x_feature]
        x = self.fc_in(x) # [1,src len, hid dim]
        for layer in self.layers:
            x, attention, KLD = layer(x)
        output = self.fc_out(x)
        return output,KLD


class EncoderLayer(nn.Module):
    def __init__(self,
                 args,
                 device,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 w1,
                 w2):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(args, hid_dim, n_heads, dropout, device, w1, w2)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(args,hid_dim,1,dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # self attention
        _src, attention, KLD = self.self_attention(src, src, src)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src, attention, KLD
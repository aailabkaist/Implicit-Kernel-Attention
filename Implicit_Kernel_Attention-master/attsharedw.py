import torch
from torch import nn
import math
import torch.nn.functional as F

class MuVarEncoder(nn.Module):  # data dependent random noise
    def __init__(self,args):
        super(MuVarEncoder, self).__init__()
        self.args = args
        self.w1 = torch.nn.Parameter(torch.empty(2*args.d_model // args.h, args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.b1 = torch.nn.Parameter(torch.empty(args.d_model // args.h,1,1),
                                             requires_grad=True)  # [dim,head,M]
        self.w2 = torch.nn.Parameter(torch.empty(args.d_model // args.h,  args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.b2 = torch.nn.Parameter(torch.empty(1,args.d_model // args.h,1,1),
                                             requires_grad=True)  # [dim,head,M]
        self.w3 = torch.nn.Parameter(torch.empty(args.d_model // args.h, args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.b3 = torch.nn.Parameter(torch.empty(1,args.d_model // args.h, 1, 1),
                                             requires_grad=True)  # [dim,head,M]

        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        torch.nn.init.zeros_(self.b1)
        torch.nn.init.zeros_(self.b2)
        torch.nn.init.zeros_(self.b3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # [b,dim,head,M]
        a,b,c,d = mu.size()
        pos_eps = torch.randn([a,b,c,self.args.M//2]).cuda() / 10.0
        neg_eps = torch.randn([a,b,c,self.args.M//2]).cuda() / 10.0

        pos_mu = mu
        neg_mu = -1 * pos_mu

        z_pos = pos_mu + std*pos_eps
        z_neg = neg_mu + std*neg_eps
        z = torch.cat([z_pos,z_neg],dim=-1)
        return z

    def forward(self, x1,x2):
        _, _, maxlen, temp_d = x1.size()
        x1 = torch.sum(x1, dim=2)/maxlen  # [batch_size,8,64*2]

        _, _, key_maxlen, temp_d = x2.size()
        x2 = torch.sum(x2, dim=2) / key_maxlen  # [batch_size,8,64*2]

        x = torch.cat([x1,x2],-1) # [batch_size,8,64*2]
        bsz, num_head, temp_d2 = x.size() # 2 * temp_d2
        x = x.transpose(2,1) # [batch_size,64*2,8]
        x = x.unsqueeze(-1) # [batch_size,64*2,8,1]

        x = x.contiguous().view(bsz, -1)  # [batch,head,maxlen,dim]
        x = x.contiguous().view(bsz, temp_d2, num_head, -1)  # [batch,dim*2,head,1]
        pre_z = F.leaky_relu(torch.einsum('bijk,il->bljk', x, self.w1) + self.b1,negative_slope=0.2) # [b,dim,head,1]
        mu = torch.einsum('bijk,il->bljk', pre_z, self.w2) + self.b2  # [b,dim,head,1]
        logvar = torch.einsum('bijk,il->bljk', pre_z, self.w3) + self.b3 # # [b,dim,head,1]
        logvar = logvar + math.log(self.args.prior_var)  # to control the variance
        z = self.reparameterize(mu,logvar) # [b,64,8,M]
        return z, mu, logvar, pre_z

class CopulaNet(nn.Module):  # data dependent random noise
    def __init__(self,args):
        super(CopulaNet, self).__init__()
        self.args = args
        self.w1 = torch.nn.Parameter(torch.empty(2*args.d_model // args.h, args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.b1 = torch.nn.Parameter(torch.empty(args.d_model // args.h,1,1),
                                             requires_grad=True)  # [dim,head,M]
        self.w2 = torch.nn.Parameter(torch.empty(args.d_model // args.h,  args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.b2 = torch.nn.Parameter(torch.empty(1,args.d_model // args.h,1,1),
                                             requires_grad=True)  # [dim,head,M]
        self.w3 = torch.nn.Parameter(torch.empty(args.d_model // args.h, args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.b3 = torch.nn.Parameter(torch.empty(1,args.d_model // args.h, 1, 1),
                                             requires_grad=True)  # [dim,head,M]

        self.copula_w1 = torch.nn.Parameter(torch.empty(args.d_model // args.h, args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.copula_b1 = torch.nn.Parameter(torch.empty(args.d_model // args.h,1,1),
                                             requires_grad=True)  # [dim,head,M]

        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.copula_w1, a=math.sqrt(5))
        torch.nn.init.zeros_(self.b1)
        torch.nn.init.zeros_(self.b2)
        torch.nn.init.zeros_(self.b3)
        torch.nn.init.zeros_(self.copula_b1)

    def forward(self, x1,x2):
        num_batch, _, maxlen, temp_d = x1.size()
        x1 = torch.sum(x1, dim=2)/maxlen  # [batch_size,8,64*2]

        _, _, key_maxlen, temp_d = x2.size()
        x2 = torch.sum(x2, dim=2) / key_maxlen  # [batch_size,8,64*2]

        x = torch.cat([x1,x2],-1) # [batch_size,8,64*2]
        bsz, num_head, temp_d2 = x.size() # 2 * temp_d2
        x = x.transpose(2,1) # [batch_size,64*2,8]
        x = x.unsqueeze(-1) # [batch_size,64*2,8,1]

        x = x.contiguous().view(bsz, -1)  # [batch,head,maxlen,dim]
        x = x.contiguous().view(bsz, temp_d2, num_head, -1)  # [batch,dim*2,head,1]
        pre_z = F.leaky_relu(torch.einsum('bijk,il->bljk', x, self.w1) + self.b1,negative_slope=0.2) # [b,dim,head,1]
        mu = torch.einsum('bijk,il->bljk', pre_z, self.w2) + self.b2  # [b,dim,head,1]
        logvar = torch.einsum('bijk,il->bljk', pre_z, self.w3) + self.b3 # # [b,dim,head,1]
        logvar = logvar + math.log(self.args.prior_var)  # to control the variance

        pre_z1 = F.tanh(torch.einsum('bijk,il->bljk', pre_z, self.copula_w1) + self.copula_b1)  # [b,dim,head,1]
        pre_z2 = pre_z1.transpose(-2,-1)
        L = torch.matmul(pre_z1, pre_z2) # [b,dim,head,head]
        L = torch.reshape(L,(num_batch*temp_d,num_head,num_head)) # [b*dim,head,head]
        L = torch.tril(L,diagonal=-1)
        I = torch.eye(num_head).cuda()
        L = L + I # [b*dim,head,head]
        return mu, logvar, L

class WNet(nn.Module):
    def __init__(self,args):
        super(WNet, self).__init__()
        self.args = args
        self.w1 = torch.nn.Parameter(torch.empty(args.d_model // args.h, args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.b1 = torch.nn.Parameter(torch.empty(args.d_model // args.h, 1,1),
                                             requires_grad=True)  # [dim,head,M]
        self.w2 = torch.nn.Parameter(torch.empty(args.d_model // args.h, args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.b2 = torch.nn.Parameter(torch.empty(args.d_model // args.h, 1,1),
                                             requires_grad=True)  # [dim,head,M]
        self.w3 = torch.nn.Parameter(torch.empty(args.d_model // args.h, args.d_model // args.h),
                                             requires_grad=True)  # [dim,head,M]
        self.b3 = torch.nn.Parameter(torch.empty(args.d_model // args.h, 1,1),
                                             requires_grad=True)  # [dim,head,M]

        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        torch.nn.init.zeros_(self.b1)
        torch.nn.init.zeros_(self.b2)
        torch.nn.init.zeros_(self.b3)

    def forward(self, z):
        w = F.leaky_relu(torch.einsum('bijk,il->bljk', torch.abs(z), self.w1) + self.b1, negative_slope=0.2)  # [64, 8, M]
        w = F.leaky_relu(torch.einsum('bijk,il->bljk', w, self.w2) + self.b2, negative_slope=0.2)  # [64, 8, M]
        w = torch.sign(z)*(torch.einsum('bijk,il->bljk', w, self.w3) + self.b3)  # [64, 8, M]
        return w

class AttSharedW(nn.Module):
    def __init__(self, args, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(AttSharedW, self).__init__()
        self.args = args
        self.d_k = d_model // h
        self.h = h
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        args.h = h
        args.d_model = d_model

        if args.att_type in ['ikandirect']: # yaglom
            self.w1 = torch.nn.Parameter(torch.empty(args.d_model // args.h, args.h, args.M),
                                    requires_grad=True)  # [dim,head,M]
            self.w2 = torch.nn.Parameter(torch.empty(args.d_model // args.h, args.h, args.M),
                                    requires_grad=True)  # [dim,head,M]
            torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        elif args.att_type in ['mikan']: # yaglom
            self.copulanet = CopulaNet(args)
            self.wnet1 = WNet(args)
            self.wnet2 = WNet(args)
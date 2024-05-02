import torch
import math
import torch.nn.functional as F

def square_dist(x1, x2, lengthscales=1.0):
    x1s = torch.sum(torch.mul(x1,x1), axis=-1)
    x2s = torch.sum(torch.mul(x2,x2), axis=-1)
    dist = -2 * torch.matmul(x1, x2.transpose(-2, -1)) # torch.Size([32, 8, 45, 45])
    dist = dist + x1s.unsqueeze(-1) + x2s.unsqueeze(-2) # [B,H,maxlen,maxlen]
    return dist

def ika_ns(x1,x2,args,d_k,w1,w2,scale,training):
    if len(w1.size()) == 3:
        phi_x11 = torch.einsum('ijkl,ljm->ijkm', x1, w1*scale) # [n,head,maxlen,M]
        phi_x12 = torch.einsum('ijkl,ljm->ijkm', x1, w2*scale) # [n,head,maxlen,M]
        phi_x21 = torch.einsum('ijkl,ljm->ijkm', x2, w1*scale) # [n,head,maxlen,M]
        phi_x22 = torch.einsum('ijkl,ljm->ijkm', x2, w2*scale) # [n,head,maxlen,M]
    elif len(w1.size()) == 4:
        phi_x11 = torch.einsum('ijkl,iljm->ijkm', x1, w1*scale) # [n,head,maxlen,M]
        phi_x12 = torch.einsum('ijkl,iljm->ijkm', x1, w2*scale) # [n,head,maxlen,M]
        phi_x21 = torch.einsum('ijkl,iljm->ijkm', x2, w1*scale) # [n,head,maxlen,M]
        phi_x22 = torch.einsum('ijkl,iljm->ijkm', x2, w2*scale) # [n,head,maxlen,M]

    phi_x1 = torch.cat([torch.cos(phi_x11)+torch.cos(phi_x12),torch.sin(phi_x11)+torch.sin(phi_x12)],dim=-1)   # [n,head,maxlen,2M]
    phi_x2 = torch.cat([torch.cos(phi_x21)+torch.cos(phi_x22),torch.sin(phi_x21)+torch.sin(phi_x22)],dim=-1)  # [n,head,maxlen,2M]

    scores = torch.matmul(phi_x1, phi_x2.transpose(-2, -1)) # [n,head,maxlen,maxlen]
    scores = scores/(4.0*args.M)
    scores = torch.mul(scores, scores)

    norm = torch.pow(torch.norm(input=x1,dim=-1,keepdim=True,p=args.p_norm),2) + \
           torch.pow(torch.norm(input=x2,dim=-1,keepdim=True,p=args.p_norm),2).transpose(-2,-1)
    norm = norm / (2*math.sqrt(d_k))
    norm = F.dropout(norm, p=args.att_dropout, training=training)  # p : drop prob.
    return scores, norm
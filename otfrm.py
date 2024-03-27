import geomloss
import torch
import torch.nn as nn

class OTLoss(nn.Module):
    def __init__(self, loss='sinkhorn', p=2, entreg=.1, cost_metric='euclidean'):
        super(OTLoss, self).__init__()
        # self.cost_func = cost_routines[cost_metric, p]
        self.loss = loss
        self.p = p
        self.entreg = entreg
        self.cost_metric = cost_metric
        # for OTNCEx
        self.otloss = geomloss.SamplesLoss(loss=self.loss, p=self.p, blur=self.entreg,#**(1/self.p)
                                            backend='tensorized', cost=self.cost_func, potentials=True)
        # for OTFRM
        self.pwdist = geomloss.SamplesLoss(loss=self.loss, p=self.p, blur=self.entreg**(1/self.p),
                                            backend='tensorized', cost=self.cost_func)
        self.frm_type = 'exact'

    def closs(self, Xs_emb, Ys, Xt_emb, Yt):
        C = self.cost_func(Xs_emb, Xt_emb)
        u, v = self.otloss(Xs_emb, Xt_emb)
        P = torch.exp(1 / self.entreg * (u.t() + v - C))  # * (pq)
        nce = self.compute_NCE(P, Ys, Yt)
        return nce

    def cotfrm(self, Xs_emb, Xt_emb):
        inter_dist = self.pwdist(Xs_emb, Xt_emb)
        # intra_dist = self.cost_func(Xs_emb, Xs_emb).mean()
        intra_dist = self.cost_func(Xt_emb, Xt_emb).mean()
        if 1 > inter_dist.item() > 0.0005:
            otfrm = (1 - intra_dist) / (1 - inter_dist)
            self.frm_type = 'exact'
        else:
            otfrm = inter_dist / intra_dist
            self.frm_type = 'approximate'
        return otfrm
    
    def cpwdist(self, Xs_emb, Xt_emb):
        return 1 / self.pwdist(Xs_emb, Xt_emb)
    
    def cost_func(self, x, y):
        if self.cost_metric=='euclidean' and self.p==1:
            return geomloss.utils.distances(x, y)
        elif self.cost_metric=='euclidean' and self.p==2:
            return geomloss.utils.squared_distances(x, y)# / 2
        else:
            # x_norm = x / x.norm(dim=-1).reshape(*x[:-1], 1)
            # y_norm = y / y.norm(dim=-1).reshape(*y[:-1], 1)
            if x.dim() == 3:
                x_norm = x / x.norm(dim=2)[:, :, None]
                y_norm = y / y.norm(dim=2)[:, :, None]
                C = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
            elif x.dim() == 2:
                x_norm = x / x.norm(dim=1)[:, None]
                y_norm = y / y.norm(dim=1)[:, None]
                C = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
            C = pow(C, self.p)# / self.p
            return C
import torch
from torch import nn
from torch.nn import functional as F

class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, device = None, alpha=0.1, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat


    def forward(self, centers):
        '''
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1
        '''
        nodevec1 = centers
        nodevec2 = centers

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
        #nodevec1 = F.normalize(self.alpha*self.lin1(nodevec1))
        #nodevec2 = F.normalize(self.alpha*self.lin2(nodevec2))

        #a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        a = torch.mm(nodevec1, nodevec1.transpose(1,0))
        #adj = F.relu(torch.tanh(self.alpha*a))
        adj = F.relu(torch.tanh(self.alpha*a))
        #adj = F.relu(self.alpha*a)
        #print(adj)
        '''
        mask = torch.zeros(centers.size(0), centers.size(0)).cuda().half() #to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        #print(s1, t1)
        #print(mask.dtype, s1.dtype)
        mask.scatter_(1,t1,s1.fill_(1))

        adj = adj*mask
        '''
        #print(adj)
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from model.dino.DinoFeaturizer import DinoFeaturizer
from utils.layer_utils import ClusterLookup
import numpy as np
import torch.distributed as dist
from utils.dist_utils import all_reduce_tensor, all_gather_tensor
from model.dag_layer import DagLayer
from matplotlib import pyplot as plt
import numpy as np
from model.graph_constructor import GraphConstructor


class CAMHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.n_classes = n_classes
        #self.class_idx = target_class
        #self.latent = nn.Conv2d(dim, 256, (1, 1))
        #self.classifier = nn.Conv2d(dim, 2, (1, 1))
        #self.classifier2 = nn.Conv2d(256, 2, (1, 1))
        self.binary_classifier = nn.Conv2d(dim, 1, (1, 1))
    def forward(self, x):
        #x = F.relu(x)
        #x = self.latent(x)
        #x = F.relu(x)
        #heatmap = self.classifier2(x)
        #heatmap = self.classifier(x)

        heatmap = self.binary_classifier(x)
        #heatmap = F.sigmoid(heatmap)
        logits = F.adaptive_avg_pool2d(heatmap, 1).squeeze()
        #logits = F.adaptive_max_pool2d(heatmap, 1).squeeze()
        return heatmap, logits

def matrix_poly(matrix, d):
    x = torch.eye(d).cuda()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)

def _h_A(A, m):
    #A = torch.ones_like(A)
    #expm_A = matrix_poly(A*A, m)
    expm_A = torch.exp(A*A)
    h_A = torch.trace(expm_A) - m
    #print(A)
    
    return h_A

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class GraphCAMHead(nn.Module):
    def __init__(self, dim, n_classes, temperature = 1):
        super().__init__()
        self.cluster_probe = ClusterLookup(dim, n_classes)
        self.temperature = temperature

        self.daglayer = DagLayer(n_classes, n_classes, i = False, initial=False)
        self.binary_classifier = nn.Conv2d(dim, 1, (1, 1))

        self.graph_constructor = GraphConstructor(n_classes, 5, dim)
        self.gcn = mixprop(dim, dim, 1, dropout = 0.3, alpha = 0.05)

    def forward(self, x, centers, inner_products = None):

        if inner_products is None:
            clusters = centers
            features = x
            inner_products = torch.einsum("bchw,nc->bnhw", features, clusters)

        A = self.graph_constructor(centers)
        inv_A = self.daglayer.inv_A


        
        dag_param = A
        h_a = _h_A(dag_param, dag_param.size()[0])
        h_a = (3*h_a + 0.5*h_a*h_a ) * 0
        
        
        data = np.abs(A.detach().cpu().numpy().copy())
        #data.resize(27,27)
        plt.figure(figsize = (10,10))
        plt.imshow(data, interpolation='nearest')
        plt.show()
        plt.savefig('A.png')
        plt.close()
        #raise SystemExit
        
      

        one_hot = F.gumbel_softmax(inner_products, tau = self.temperature, hard = True, dim=1)
        z_q = torch.einsum('b n h w, n d -> b d h w', one_hot, centers)
        heatmap = self.binary_classifier(z_q)
        logits = F.adaptive_avg_pool2d(heatmap, 1).squeeze()
        print(inner_products, logits)

        

        new_centers = self.gcn(new_centers, A)
        normed_inner_products = torch.einsum("bchw,nc->bnhw", features, new_centers)

        cluster_probs = F.one_hot(torch.argmax(normed_inner_products, dim=1), centers.shape[0]) \
            .permute(0, 3, 1, 2).to(torch.float32)
        
        cluster_loss = -(cluster_probs * F.softmax(normed_inner_products, dim = 1)).sum(1).mean()
        #cluster_loss = F.cross_entropy(inner_products, cluster_probs)
        
        
        return heatmap, logits, l_m + cluster_loss, inner_products

class CausalCAMHead(nn.Module):
    def __init__(self, dim, n_classes, temperature = 1):
        super().__init__()
        self.cluster_probe = ClusterLookup(dim, n_classes)
        self.temperature = temperature

        self.daglayer = DagLayer(n_classes, n_classes, i = False, initial=False)
        self.binary_classifier = nn.Conv2d(dim, 1, (1, 1))

        self.graph_constructor = GraphConstructor(n_classes, 5, dim)

    def forward(self, x, centers, inner_products = None):

        if inner_products is None:

            clusters = centers
            features = x
            normed_clusters = F.normalize(centers, dim=1)
            normed_features = F.normalize(x, dim=1)
            inner_products = torch.einsum("bchw,nc->bnhw", features, clusters)
            normed_inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
            #one_hot = gumbel_softmax(inner_products, temperature = self.temperature, hard = True)

        #else:
        #    inner_products *= 10

        A = self.graph_constructor(centers)
        inv_A = self.daglayer.inv_A

        
        dag_param = A
        h_a = _h_A(dag_param, dag_param.size()[0])
        h_a = (3*h_a + 0.5*h_a*h_a ) * 0
        
        
        data = np.abs(A.detach().cpu().numpy().copy())
        #data.resize(27,27)
        plt.figure(figsize = (10,10))
        plt.imshow(data, interpolation='nearest')
        plt.show()
        plt.savefig('A.png')
        plt.close()
        #raise SystemExit
        
        

        #print(inner_products.size())
        b = inner_products.size(0)
        n = inner_products.size(1)
        h = inner_products.size(2)
        w = inner_products.size(3)
        inner_products = inner_products.permute(0, 2, 3, 1)
        inner_products = inner_products.reshape(b * h * w, -1)   
        
        #inner_products_d = inner_products.detach()

        inner_products = self.daglayer.calculate_dag(inner_products, inv_A)
        inner_products_m = self.daglayer.mask_z(inner_products.unsqueeze(-1), A)
        #print(inner_products.size(), inner_products_m.size())

        #l_m = F.mse_loss(inner_products, inner_products_m.squeeze()) * 0
        l_m = F.mse_loss((self.daglayer.I - A.t()).matmul(inv_A), self.daglayer.I)
        print(l_m)
        

        inner_products = inner_products_m
        inner_products = inner_products.reshape(b, h, w, n)
        inner_products = inner_products.permute(0, 3, 1, 2)
       

        one_hot = F.gumbel_softmax(inner_products, tau = self.temperature, hard = True, dim=1)
        z_q = torch.einsum('b n h w, n d -> b d h w', one_hot, centers)
        heatmap = self.binary_classifier(z_q)
        logits = F.adaptive_avg_pool2d(heatmap, 1).squeeze()
        print(inner_products, logits)

        
        normed_inner_products = normed_inner_products.permute(0, 2, 3, 1)
        normed_inner_products = normed_inner_products.reshape(b * h * w, -1)   
        normed_inner_products = self.daglayer.calculate_dag(normed_inner_products, inv_A)
        normed_inner_products = self.daglayer.mask_z(normed_inner_products.unsqueeze(-1), A)
        normed_inner_products = normed_inner_products.reshape(b, h, w, n)
        normed_inner_products = normed_inner_products.permute(0, 3, 1, 2)

        cluster_probs = F.one_hot(torch.argmax(normed_inner_products, dim=1), centers.shape[0]) \
            .permute(0, 3, 1, 2).to(torch.float32)
        
        cluster_loss = -(cluster_probs * F.softmax(normed_inner_products, dim = 1)).sum(1).mean()
        #cluster_loss = F.cross_entropy(inner_products, cluster_probs)
        
        
        return heatmap, logits, l_m + cluster_loss, inner_products

class VQClusterHead(nn.Module):
    def __init__(self, dim, n_classes, temperature = 1):
        super().__init__()
        self.cluster_probe = ClusterLookup(dim, n_classes)
        self.temperature = temperature

    def forward(self, x, centers, inner_products = None):

        if inner_products is None:

            normed_clusters = centers
            normed_features = x

            inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
            #one_hot = gumbel_softmax(inner_products, temperature = self.temperature, hard = True)

        #else:
        #    inner_products *= 10
        
        one_hot = F.gumbel_softmax(inner_products, tau = self.temperature, hard = True, dim=1)

        #one_hot = F.one_hot(torch.argmax(inner_products, dim = 1), centers.size(0)).permute(0, 3, 1, 2).float()
        #one_hot = F.softmax(inner_products)
        
        #print((one_hot != one_hot_max).sum())
        #print(one_hot_max.size(), torch.argmax(one_hot_max[0].sum(2).sum(1)))
        
        #print(centers.size())
        #print(one_hot.size())
        z_q = torch.einsum('b n h w, n d -> b d h w', one_hot, centers)
        cluster_loss, cluster_preds, cluster_logits = self.cluster_probe(z_q, None, is_direct=False)
        return cluster_loss, cluster_preds, cluster_logits, z_q



class VQCAMHead(nn.Module):
    def __init__(self, dim, n_classes, temperature = 1):#10):
        super().__init__()
        #self.n_classes = n_classes
        #self.class_idx = target_class
        #self.latent = nn.Conv2d(dim, 256, (1, 1))
        #self.classifier = nn.Conv2d(dim, 2, (1, 1))
        #self.classifier2 = nn.Conv2d(256, 2, (1, 1))
        #self.out_feat = nn.Conv2d(n_classes, dim, (1, 1))
        self.binary_classifier = nn.Conv2d(dim, 1, (1, 1))
        #self.att = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.temperature = temperature
        
    def forward(self, x, centers, inner_products = None):
        #x = F.relu(x)
        #x = self.latent(x)
        #x = F.relu(x)
        #heatmap = self.classifier2(x)
        #heatmap = self.classifier(x)

        #centers = centers.clone().detach()
        '''
        normed_clusters = F.normalize(centers, dim=1).unsqueeze(0).repeat(x.size(0), 1, 1)
        normed_features = F.normalize(x, dim=1).permute(0, 2, 3, 1)
        normed_features = normed_features.reshape(x.size(0), -1, x.size(1))
        print(normed_clusters.size(), normed_features.size())
        
        z_q = self.att(normed_features, normed_clusters , normed_clusters)
        
        '''
        if inner_products is None:
            #normed_clusters = F.normalize(centers, dim=1)
            #normed_features = F.normalize(x, dim=1)
            normed_clusters = centers
            normed_features = x

            inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
            #one_hot = gumbel_softmax(inner_products, temperature = self.temperature, hard = True)
        #else:
        #    inner_products *= 10
        
        
        one_hot = F.gumbel_softmax(inner_products, tau = self.temperature, hard = True, dim=1)
        #one_hot = F.one_hot(torch.argmax(inner_products, dim = 1), centers.size(0)).permute(0, 3, 1, 2).float()
        #one_hot = F.softmax(inner_products)
        
        #print((one_hot != one_hot_max).sum())
        #print(one_hot_max.size(), torch.argmax(one_hot_max[0].sum(2).sum(1)))
        
        #print(centers.size())
        #print(one_hot.size())
        z_q = torch.einsum('b n h w, n d -> b d h w', one_hot, centers)
        #print(one_hot[:, 0].size(), one_hot.size())
        #print(z_q[0])
        #z_q = torch.einsum('b n h w, n d -> b d h w', inner_products, centers)
        #print(z_q[0])
        #z_q = self.out_feat(one_hot)
        #print(z_q[0])
        
        heatmap = self.binary_classifier(z_q)
        #heatmap = F.sigmoid(heatmap / self.temperature)
        logits = F.adaptive_avg_pool2d(heatmap, 1).squeeze()
        #logits = F.adaptive_max_pool2d(heatmap, 1).squeeze()

        #print(one_hot_max[:].unsqueeze(1).sum())
        #raise SystemExit


        return heatmap, logits #one_hot[:, 93].unsqueeze(1).float(), logits #heatmap, logits



class STEGOmodel(nn.Module):
    # opt["model"]
    def __init__(self,
                 opt: dict,
                 n_classes:int
                 ):
        super().__init__()
        self.opt = opt
        self.n_classes= n_classes


        if not opt["continuous"]:
            dim = n_classes
        else:
            dim = opt["dim"]

        if opt["arch"] == "dino":
            self.net = DinoFeaturizer(dim, opt)
        else:
            raise ValueError("Unknown arch {}".format(opt["arch"]))

        self.cluster_probe = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.cluster_probe2 = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe2 = nn.Conv2d(dim, n_classes, (1, 1))

        self.cluster_probe3 = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe3 = nn.Conv2d(dim, n_classes, (1, 1))

        self.cam = GraphCAMHead(dim, n_classes + opt["extra_clusters"])
        self.sup_cluster = VQClusterHead(dim, n_classes + opt["extra_supclusters"])
        self.supcluster_cam = VQCAMHead(dim, n_classes + opt["extra_supclusters"])

    def forward(self, x: torch.Tensor):
        return self.net(x)[1]

    @classmethod
    def build(cls, opt, n_classes):
        # opt = opt["model"]
        m = cls(
            opt = opt,
            n_classes= n_classes
        )
        print(f"Model built! #params {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count


if __name__ == '__main__':
    net = STEGOmodel()
    dummy_input = torch.empty(2, 3, 352, 1216)
    dummy_output = net(dummy_input)[0]
    print(dummy_output.shape)  # (2, 1, 88, 304)


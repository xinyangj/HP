from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from model.dino.DinoFeaturizer import DinoFeaturizer
from utils.layer_utils import ClusterLookup
import numpy as np
import torch.distributed as dist
from utils.dist_utils import all_reduce_tensor, all_gather_tensor


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

class VQCAMHead(nn.Module):
    def __init__(self, dim, n_classes, temperature = 10):
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
        
    def forward(self, x, centers):
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

        #normed_clusters = F.normalize(centers, dim=1)
        #normed_features = F.normalize(x, dim=1)
        normed_clusters = centers
        normed_features = x

        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        #one_hot = gumbel_softmax(inner_products, temperature = self.temperature, hard = True)
        #one_hot = F.gumbel_softmax(inner_products, tau = self.temperature, hard = True, dim=1)
        one_hot = F.one_hot(torch.argmax(inner_products, dim = 1), centers.size(0)).permute(0, 3, 1, 2).float()
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

        self.cam = VQCAMHead(dim, n_classes + opt["extra_clusters"])
        


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


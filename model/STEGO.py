from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from model.dino.DinoFeaturizer import DinoFeaturizer
from utils.layer_utils import ClusterLookup
import numpy as np
import torch.distributed as dist
from utils.dist_utils import all_reduce_tensor, all_gather_tensor
from model.graph_constructor import GraphConstructor
from torch_geometric.nn.conv import MixHopConv, GCNConv
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

class CustomMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=True):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first=batch_first)
        self.batch_first = batch_first

    def kq_score(self, query, key):
        if self.batch_first:
            batch_size, seq_length_q, embed_dim = query.size()
            batch_size_k, seq_length_k, embed_dim_k = key.size()
            assert batch_size == batch_size_k, "Batch sizes of query and key must match"
        else:
            seq_length_q, batch_size, embed_dim = query.size()
            seq_length_k, batch_size_k, embed_dim_k = key.size()
            assert batch_size == batch_size_k, "Batch sizes of query and key must match"
        
        # Step 1: Linear projections
        q_proj_weight = self.in_proj_weight[:self.embed_dim, :]
        k_proj_weight = self.in_proj_weight[self.embed_dim:2*self.embed_dim, :]
        
        if self.in_proj_bias is not None:
            q_proj_bias = self.in_proj_bias[:self.embed_dim]
            k_proj_bias = self.in_proj_bias[self.embed_dim:2*self.embed_dim]
        else:
            q_proj_bias = k_proj_bias = None
        
        Q = nn.functional.linear(query, q_proj_weight, q_proj_bias)
        K = nn.functional.linear(key, k_proj_weight, k_proj_bias)
        
        # Step 2: Reshape Q, K
        if self.batch_first:
            Q = Q.contiguous().view(batch_size, seq_length_q, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.contiguous().view(batch_size, seq_length_k, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            Q = Q.contiguous().view(seq_length_q, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            K = K.contiguous().view(seq_length_k, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Step 3: Compute raw attention scores
        raw_attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        
        # Step 4: Normalize scores for each head separately
        raw_attention_scores = raw_attention_scores.view(batch_size, self.num_heads, seq_length_q, seq_length_k)
        min_scores = raw_attention_scores.min(dim=-1, keepdim=True)[0]
        max_scores = raw_attention_scores.max(dim=-1, keepdim=True)[0]
        normalized_attention_scores = (raw_attention_scores - min_scores) / (max_scores - min_scores + 1e-6)  # Add epsilon to avoid division by zero
        
        # Step 5: Average over heads
        mean_attention_scores = normalized_attention_scores.mean(dim=1)
        
        return mean_attention_scores

class VQAttCAMHead(nn.Module):
    def __init__(self, dim, n_classes, temperature = 10):
        super().__init__()
        #self.n_classes = n_classes
        #self.class_idx = target_class
        #self.latent = nn.Conv2d(dim, 256, (1, 1))
        #self.classifier = nn.Conv2d(dim, 2, (1, 1))
        #self.classifier2 = nn.Conv2d(256, 2, (1, 1))
        #self.out_feat = nn.Conv2d(n_classes, dim, (1, 1))
        #self.binary_classifier = nn.Conv2d(dim, 1, (1, 1))

        self.binary_classifier0 = nn.Conv2d(dim, 1, (1, 1))
        self.binary_classifier1 = nn.Conv2d(dim, 1, (1, 1))
        self.att = CustomMultiheadAttention(dim, 8, batch_first=True)
        self.cross_att = CustomMultiheadAttention(dim, 8, batch_first=True)

        #self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.ffn0 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.ffn1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        #self.att = nn.MultiheadAttention(dim, 1, batch_first=True)

        #self.norm = nn.LayerNorm(dim)
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.temperature = temperature
        self.cls_token = nn.Parameter(torch.randn(1,1, dim))
    
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

        normed_clusters = centers
        normed_features = x

        heatmap0 = self.binary_classifier0(x)
        logits0 = F.adaptive_avg_pool2d(heatmap0, 1).squeeze()

        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        one_hot = F.gumbel_softmax(inner_products, tau = self.temperature, hard = True, dim=1)
        z_q = torch.einsum('b n h w, n d -> b d h w', one_hot, centers)
        heatmap = self.binary_classifier0(z_q)
        logits = F.adaptive_avg_pool2d(heatmap, 1).squeeze()
        
        z_q = z_q.permute((0, 2, 3, 1)).reshape(x.size(0), x.size(2) * x.size(3), -1)

        '''
        #z_q = torch.cat([cls_tokens, z_q], dim=1)
        z_q_att, a = self.att(z_q, z_q, z_q)
        z_q = z_q_att + z_q
        z_q = self.norm(z_q)
        z_q = self.ffn(z_q)
        '''
        
        z_q_att, a = self.att(z_q, z_q, z_q)
        z_q = z_q_att + z_q
        z_q = self.norm0(z_q)
        z_q = self.ffn0(z_q)

        cls_tokens = self.cls_token.repeat(z_q.size(0), 1, 1)
        z_q_out, a_cross = self.cross_att(cls_tokens, z_q, z_q)
        supcluster_heatmap = self.cross_att.kq_score(cls_tokens, z_q)
        supcluster_heatmap = supcluster_heatmap.squeeze().reshape(x.size(0), x.size(2), x.size(3))
        supcluster_heatmap = supcluster_heatmap.unsqueeze(1)
        
        #z_q = z_q + z_q_att
        z_q = self.norm1(z_q_out)
        z_q = self.ffn1(z_q)

        cls_tokens = z_q[:, 0]
        #print(cls_tokens.size())
        supcluster_logits = self.binary_classifier1(cls_tokens.unsqueeze(-1).unsqueeze(-1))
        supcluster_logits = supcluster_logits.squeeze()
        #supcluster_heatmap = a_cross[:, 0].reshape(x.size(0), x.size(2), x.size(3))
        #supcluster_heatmap = supcluster_heatmap.unsqueeze(1)

        cls_tokens = self.cls_token.repeat(z_q.size(0), 1, 1)
        

        '''
        z_q = z_q.reshape(x.size(0), x.size(2), x.size(3), -1).permute(0, 3, 1, 2)    
        heatmap = self.binary_classifier(z_q)
        logits = F.adaptive_avg_pool2d(heatmap, 1).squeeze()
        '''

        return heatmap, logits, supcluster_heatmap, supcluster_logits, a, logits0 #one_hot[:, 93].unsqueeze(1).float(), logits #heatmap, logits


class ClusterOnGNN(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super(ClusterOnGNN, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.bn = torch.nn.BatchNorm2d(dim)
        self.clusters = None
        self.lin = nn.Conv2d(dim, dim, (1, 1))
        self.graph_constructor = GraphConstructor(n_classes, 5, dim)
        self.gcn = GCNConv(dim, dim, bias = False)

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, centers, alpha, log_probs=False, is_direct=False):
        A = self.graph_constructor(centers)
        edge_index = A.nonzero().t().contiguous()
        edge_weight = A[edge_index[0], edge_index[1]]

        data = np.abs(A.detach().cpu().numpy().copy())
        plt.figure(figsize = (10,10))
        plt.imshow(data, interpolation='nearest')
        plt.show()
        plt.savefig('A.png')
        plt.close()

        fig, ax = plt.subplots(figsize = (10,10))
        selector = [10, 29, 3, 44, 46, 53, 69]
        data = (A[selector][:, selector] + A[selector][:, selector].t())
        #ax = plt.gca()
        
        #plt.figure()
        plt.imshow(data.detach().cpu().numpy().copy(), interpolation='nearest')
        ax.set_xticklabels(['', 'sky', 'road', 'vehicle', 'car body', 'window', 'wheel', 'grass'])
        ax.set_yticklabels(['', 'sky', 'road', 'vehicle', 'car body', 'window', 'wheel', 'grass'])
        plt.show()
        plt.savefig('a.png')
        plt.close()

        self.clusters = self.gcn(centers, edge_index, edge_weight)
        #features = self.lin(x)
        #features = self.bn(features)
        

        if is_direct:
            inner_products = x
        else:
            normed_clusters = F.normalize(self.clusters, dim=1)
            normed_features = F.normalize(x, dim=1)
            inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()

        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, inner_products #cluster_probs
    

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

        self.cam = VQAttCAMHead(dim, n_classes + opt["extra_clusters"])
        self.sup_cluster = ClusterOnGNN(dim, n_classes + opt["extra_clusters"])
        self.supcluster_cam = VQCAMHead(dim, n_classes + opt["extra_clusters"])
        

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


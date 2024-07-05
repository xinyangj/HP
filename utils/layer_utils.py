import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graph_constructor import GraphConstructor
from torch_geometric.nn.conv import MixHopConv, GCNConv


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

        self.graph_constructor = GraphConstructor(n_classes, 5, dim)
        self.gcn = GCNConv(dim, dim, bias = False)

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False, is_direct=False, mp = True):
        if mp:                
            A = self.graph_constructor(self.clusters)
            edge_index = A.nonzero().t().contiguous()
            edge_weight = A[edge_index[0], edge_index[1]]
            clusters = self.gcn(self.clusters, edge_index, edge_weight)
        else:
            clusters = self.clusters

        if is_direct:
            inner_products = x
        else:
            normed_clusters = F.normalize(clusters, dim=1)
            normed_features = F.normalize(x, dim=1)
            inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        
        p = torch.softmax(torch.einsum("bchw,nc->bnhw", x, self.clusters), dim=1)
        # Compute the entropy
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=1)
        # Return the mean entropy
        entropy = torch.mean(entropy)
        #print(entropy)
        #cluster_loss += - 0.1 * entropy



        '''
        cluster_centers = F.normalize(self.clusters, dim=1)
        similarity_matrix = torch.mm(cluster_centers, cluster_centers.t())  # Cosine similarity matrix
        # Exclude diagonal elements (similarity of a center with itself)
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)
        center_sim = torch.mean(similarity_matrix)
        print(center_sim)
        cluster_loss += center_sim
        '''
        
        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, inner_products, clusters #cluster_probs

    def forward_entropy(self, x, alpha, log_probs=False, is_direct=False):
        if is_direct:
            inner_products = x
        else:
            normed_clusters = F.normalize(self.clusters, dim=1)
            normed_features = F.normalize(x, dim=1)
            inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        
        p = torch.softmax(torch.einsum("bchw,nc->bnhw", x, self.clusters), dim=1)
        # Compute the entropy
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=1)
        # Return the mean entropy
        entropy = torch.mean(entropy)
        #print(entropy)
        return -1 * entropy




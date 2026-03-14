# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import GCNConv, GATConv 
from torch_geometric.data import Data
# from torch_geometric.utils import negative_sampling #, train_test_split_edges
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import networkx as nx
import random
import PyWGCNA
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class GetLoader(Dataset):
    # to extract different 20-column-graphSets from source01/target matrix 
    def __init__(self, df1, df2, label=1, rst=0.8, samplen=1000, seed=1234, scaled=False, soft=False, max_repeats = [10,10]):
        # df: pandas dataframe. <source01/target01> gene*sample
        # df1-df2: df of normal/cancer  OR  treated-cellline/cellline
        # label: domain label - tissue/cell 0/1. 
        super(GetLoader, self).__init__()
        random.seed(seed)  # Set seed for all random operations
        if scaled:
            scaler1 = preprocessing.StandardScaler()
            self.df1 = pd.DataFrame(scaler1.fit_transform(df1.values), index=df1.index, columns=df1.columns)
            scaler2 = preprocessing.StandardScaler()
            self.df2 = pd.DataFrame(scaler2.fit_transform(df2.values), index=df2.index, columns=df2.columns)

        else:
            self.df1 = df1; self.df2 = df2
        
        self.rst = rst # for edge threshold
        self.soft = soft
        self.label = label
        self.samplen = samplen
        self.comb1 = self.pickSample(df1, max_repeats[0])  # take 20 samples*samplen times
        self.comb2 = self.pickSample(df2, max_repeats[1])   
        
    def pickSample(self, df, max_repeats):
        total_numbers = df.shape[1] 
        num_groups = self.samplen
        group_size = 20
        
        numbers = list(range(total_numbers))
        groups = []

        while len(groups) < num_groups:
            current_group = set(random.sample(numbers, group_size))
            if all(len(current_group.intersection(existing_group)) <= max_repeats for existing_group in groups):
                groups.append(list(current_group))

        return groups
    
    
    def __getitem__(self, item):
        comb_index1 = self.comb1[item]; g1 = self.df1.iloc[:, comb_index1]  # gene * sample
        comb_index2 = self.comb2[item]; g2 = self.df2.iloc[:, comb_index2]

        if self.soft:
            edge_index_1 = self.makeWGCNA(g1.T)
            edge_index_2 = self.makeWGCNA(g2.T)    
        else:
            edge_index_1 = self.makeGraph(g1.T)   # 输入sample*gene
            edge_index_2 = self.makeGraph(g2.T)

        data_1 = Data(x = torch.tensor(g1.values, dtype=torch.float32), edge_index=edge_index_1, t=self.label)
        data_2 = Data(x = torch.tensor(g2.values, dtype=torch.float32), edge_index=edge_index_2, t=self.label)
        # delta = Data(x = torch.rand(g1.shape[0],2), edge_index=delta, t=3) # 只是为了edge_index能顺利concat,x,label等值是占位作用
        
        return data_1, data_2 
    
    def __len__(self):
        return len(self.comb1)
    
    def makeGraph(self, df):
        # input: sample*gene
        # len(a[abs(a)>0.7])/(948*947)  判断rst合适值
        rst = self.rst
        cor = df.corr()   #a = cor1.values; len(a[abs(a)>rst])/(948*947)
        a = (abs(cor) > rst) * cor/abs(cor); np.fill_diagonal(a.values, 0)
        # a = a2.values; len(a[abs(a)>0])/(948*947)
       
        g = abs(a)
        g = nx.from_numpy_array(g.values)
        adj = nx.to_scipy_sparse_array(g).tocoo()
        row = torch.from_numpy(adj.row)
        col = torch.from_numpy(adj.col)
        edge_index= torch.stack([row, col], dim=0)
        return edge_index
    
    def makeWGCNA(self, df, type='unsigned'):
        # input: sample*gene   return:gene*gene
        # import PyWGCNA
        rst = self.rst
        w = PyWGCNA.WGCNA(name='geneTOM', geneExp=df)
        power = 6
        if type=='signed':
            adjacency = ((1 + np.corrcoef(w.geneExpr.to_df().T))/2 ) ** power
        elif type=='unsigned':
            adjacency = abs(np.corrcoef(w.geneExpr.to_df().T)) ** power # if type=='unsigned'
        tom = w.TOMsimilarity(adjacency, TOMType=type).values
        np.fill_diagonal(tom, 0)
        adj = np.where(tom >= rst, 1, 0) 
        g = nx.from_numpy_array(adj)
        adj = nx.to_scipy_sparse_array(g).tocoo()
        row = torch.from_numpy(adj.row)
        col = torch.from_numpy(adj.col)
        edge_index= torch.stack([row, col], dim=0)
        return edge_index
    

####################### 测试模型架构 
# Gradient Reversal Layer
class ReverseLayerF(Function):     
    @staticmethod
    def forward(ctx, x, alpha): # ctx必须是第一个参数,可储存tensor供backward用
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):  # grad_output储存forward后tensor的梯度.return的是每个(forward)input的梯度
        output = grad_output.neg() * ctx.alpha
        return output, None

# Separate Encoder (for cell-line or tissue-specific features)


#graph = T.RandomNodeDrop(p=0.1)(graph)

class SeparateEncoder(nn.Module):
    def __init__(self, input_dim=20, output_dim=32, gnn='gcn', gnnLayer=3, dropoutRate=0, nhead=2):
        """
        GNN-based Separate Encoder for domain-specific embeddings.
        
        Args:
            input_dim (int): Dimensionality of input features.
            output_dim (int): Dimensionality of output embeddings.
            gnn (str): GNN type ('gcn' or 'gat').
            gnnLayer (int): Number of GNN layers.
            dropoutRate (float): Dropout rate applied after each layer.
            nhead (int): Number of attention heads (for GAT only).
        """
        super(SeparateEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.gnnLayer = gnnLayer
        self.dropoutRate = dropoutRate

        # Construct layers based on the GNN type
        self.construct_gnn_layers(gnn)

        # Initialize parameters
        self.reset_parameters()
        
        print("Constructed %s Layers for Separate Encoder:"%str.upper(gnn))
        for idx, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            print(f"Layer {idx + 1}: {layer}")
            print(f"Activation {idx + 1}: {activation}")
        
    def construct_gnn_layers(self, gnn):
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        input_dims = [self.input_dim, 64, 48][:self.gnnLayer]
        output_dims = input_dims[1:] + [self.output_dim]

        for in_dim, out_dim in zip(input_dims, output_dims):
            if gnn == 'gat':
                self.layers.append(GATConv(in_dim, out_dim, heads=self.nhead if out_dim != self.output_dim else 1))
                self.activations.append(
                    nn.Sequential(
                        nn.BatchNorm1d(self.nhead * out_dim if out_dim != self.output_dim else out_dim),
                        nn.Dropout(self.dropoutRate),
                        nn.ELU() if out_dim != self.output_dim else nn.Identity()
                    )
                )
            else:  # Default to GCN
                self.layers.append(GCNConv(in_dim, out_dim))
                self.activations.append(
                    nn.Sequential(
                        nn.BatchNorm1d(out_dim),
                        nn.Dropout(self.dropoutRate),
                        nn.ReLU() if out_dim != self.output_dim else nn.Identity()
                    )
                )
    
    def reset_parameters(self):
        """
        Initialize parameters for GNN layers and BatchNorm layers.
        """
        for layer in self.layers:
            if isinstance(layer, (GCNConv, GATConv)):
                layer.reset_parameters()  # Use PyG's built-in parameter initialization for GNN layers
        
        for activation in self.activations:
            for submodule in activation:
                if isinstance(submodule, nn.BatchNorm1d):
                    submodule.reset_parameters()  # Reset BatchNorm parameters

    def encode(self, x, edge_index):
        feature = x
        for layer, activation in zip(self.layers, self.activations):
            feature = activation(layer(feature, edge_index))
        return feature

    def forward(self, input_data):
        x, edge_index = input_data.x, input_data.edge_index
        return self.encode(x, edge_index) 




# Shared Encoder (for shared embedding extraction)
"""
## separate -> shared
class SharedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead=2, dropoutRate=0):
        super(SharedEncoder, self).__init__()

        self.gnn = GATConv(input_dim, hidden_dim, heads=nhead)
        self.activation = nn.Sequential(nn.BatchNorm1d(hidden_dim*nhead),
                            nn.ELU(),
                            nn.Dropout(dropoutRate),
                            nn.Linear(hidden_dim*nhead, output_dim))                   
        
    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        x = self.activation(x)
        return x
"""
## independent encoders 
class SharedEncoder(nn.Module):
    def __init__(self, input_dim=20, output_dim=10, gnn='gat', gnnLayer=2, dropoutRate=0, nhead=2):
        super(SharedEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.gnnLayer = gnnLayer
        self.dropoutRate = dropoutRate

        # Construct layers based on the GNN type
        self.construct_gnn_layers(gnn)

        # Initialize parameters
        self.reset_parameters()
        print("Constructed %s Layers for Shared Encoder:"%str.upper(gnn))
        for idx, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            print(f"Layer {idx + 1}: {layer}")
            print(f"Activation {idx + 1}: {activation}")
        
        
    def construct_gnn_layers(self, gnn):
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        input_dims = [self.input_dim, 64, 32][:self.gnnLayer]
        output_dims = input_dims[1:] + [self.output_dim]
        input_dims = [self.input_dim] + [2*i for i in input_dims[1:]]

        for in_dim, out_dim in zip(input_dims, output_dims):
            if gnn == 'gat':
                self.layers.append(GATConv(in_dim, out_dim, heads=self.nhead))
                self.activations.append(
                    nn.Sequential(
                        nn.BatchNorm1d(self.nhead * out_dim),
                        nn.Dropout(self.dropoutRate),
                        nn.ELU()
                    )
                )
                
            else:  # Default to GCN
                self.layers.append(GCNConv(in_dim, out_dim))
                self.activations.append(
                    nn.Sequential(
                        nn.BatchNorm1d(out_dim),
                        nn.Dropout(self.dropoutRate),
                        nn.ReLU() if out_dim != self.output_dim else nn.Identity()
                    )
                )
        if gnn == 'gat':
            self.layers.append(nn.Linear(self.output_dim * self.nhead, self.output_dim))
            self.activations.append(nn.Sequential(nn.Identity()))
    
    def reset_parameters(self):
        """
        Initialize parameters for GNN layers and BatchNorm layers.
        """
        for layer in self.layers:
            if isinstance(layer, (GCNConv, GATConv)):
                layer.reset_parameters()  # Use PyG's built-in parameter initialization for GNN layers
        
        for activation in self.activations:
            for submodule in activation:
                if isinstance(submodule, nn.BatchNorm1d):
                    submodule.reset_parameters()  # Reset BatchNorm parameters

    def encode(self, x, edge_index):
        feature = x
        for layer, activation in zip(self.layers, self.activations):
            if isinstance(layer, nn.Linear):
                feature = activation(layer(feature))
            else:
                feature = activation(layer(feature, edge_index))
        return feature

    def forward(self, input_data):
        x, edge_index = input_data.x, input_data.edge_index
        return self.encode(x, edge_index) 
""" test
# Example input
node_features = torch.randn(50, 20)  # 50 nodes, 20 features each
edge_index = torch.randint(0, 50, (2, 150))  # 150 edges

# Initialize the shared encoder
shared_encoder = SharedEncoder(input_dim=20, hidden_dim=16, output_dim=8, nhead=2, dropoutRate=0.1)

# Forward pass
shared_embedding = shared_encoder(node_features, edge_index)
print("Output shape:", shared_embedding.shape)
"""


# Decoder (for reconstruction)
class Decoder(nn.Module):
    # ExpressionMatrixDecoder
    def __init__(self, input_dim, hidden_dim, output_dim, dropoutRate=0.1):
        super(Decoder, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropoutRate)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            # nn.Sigmoid()  # Optional: Use sigmoid if input is normalized between 0 and 1
        )

    def forward(self, embedding):
        hidden = self.hidden_layer(embedding)
        reconstructed_matrix = self.output_layer(hidden)
        return reconstructed_matrix
# criterion = nn.MSELoss()
# loss = criterion(reconstructed_matrix, original_matrix)

# Discriminator (for domain confusion)
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domains=2, dropoutRate=0.5):
        super(Discriminator, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),  # Normalize hidden layer
            nn.Dropout(dropoutRate),     # Add regularization
        )
        
        self.output_layer = nn.Linear(hidden_dim, num_domains)  # Map to domain predictions

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x: Input embedding (batch_size x input_dim).

        Returns:
            logits: Logits for domain classification (batch_size x num_domains).
        """
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)  # Raw logits for classification
        return logits
        
# reversed_embedding = ReverseLayerF.apply(embedding, alpha)
# logits = Discriminator(reversed_embedding)
# criterion = nn.CrossEntropyLoss()  # This expects raw logits (no activation) 集成了softmax操作,类似BCEWithLogitsLoss对sigmoid的集成
# loss = criterion(logits, domain_labels)


# Full Model
class DrugRepurposingModel(nn.Module):
    def __init__(self, input_dim=20, shared_dim=8, separate_dim=32, hidden_decoder_dim=32, 
                 hidden_desc_dim = 5, nlayer_separate=2, nlayer_shared=2, dp=0, nhead=2):
        super(DrugRepurposingModel, self).__init__()

        # Separate encoders
        self.cellline_encoder = SeparateEncoder(input_dim=input_dim, output_dim=separate_dim, gnn='gcn', gnnLayer=nlayer_separate, dropoutRate=dp)
        self.tissue_encoder = SeparateEncoder(input_dim=input_dim, output_dim=separate_dim, gnn='gcn', gnnLayer=nlayer_separate, dropoutRate=dp)

        # Shared encoder
        self.shared_encoder = SharedEncoder(input_dim=input_dim, output_dim=shared_dim, gnn='gat', gnnLayer=nlayer_shared, nhead=nhead, dropoutRate=dp)

        # Decoders
        self.cellline_decoder = Decoder(shared_dim + separate_dim, hidden_decoder_dim, input_dim, dropoutRate=dp)
        self.tissue_decoder = Decoder(shared_dim + separate_dim, hidden_decoder_dim, input_dim, dropoutRate=dp)

        # Discriminator
        self.discriminator = Discriminator(shared_dim, hidden_desc_dim, dropoutRate=dp)

    def forward(self, tissue_normal, tissue_cancer,cellline_trt, cellline_dmso, alpha):
        # Separate embeddings
        cellline_separate_dmso = self.cellline_encoder(cellline_dmso)
        cellline_separate_trt = self.cellline_encoder(cellline_trt)
        
        tissue_separate_cancer = self.tissue_encoder(tissue_cancer)
        tissue_separate_normal = self.tissue_encoder(tissue_normal)

        # Shared embeddings
        cellline_shared_dmso = self.shared_encoder(cellline_dmso)
        cellline_shared_trt = self.shared_encoder(cellline_trt)
        
        tissue_shared_cancer = self.shared_encoder(tissue_cancer)
        tissue_shared_normal = self.shared_encoder(tissue_normal)
        
        batch_size = len(tissue_normal.t)
        n_genes = tissue_normal.ptr[1].item() if batch_size > 1 else len(tissue_normal.x)
        shared_embedding = torch.cat([torch.mean(tissue_shared_cancer.view(batch_size, n_genes, -1), dim=1),   # shape will be [shared_dim]
                                      torch.mean(tissue_shared_normal.view(batch_size, n_genes, -1), dim=1),
                                      torch.mean(cellline_shared_dmso.view(batch_size, n_genes, -1), dim=1),
                                      torch.mean(cellline_shared_trt.view(batch_size, n_genes, -1), dim=1)], dim=0)


        # Apply Gradient Reversal Layer for adversarial training
        reversed_embedding = ReverseLayerF.apply(shared_embedding, alpha)
        domain_preds = self.discriminator(reversed_embedding)

        # Reconstruction
        cellline_reconstructed_dmso = self.cellline_decoder(torch.cat([cellline_separate_dmso, cellline_shared_dmso], dim=1))
        cellline_reconstructed_trt = self.cellline_decoder(torch.cat([cellline_separate_trt, cellline_shared_trt], dim=1))
        tissue_reconstructed_normal = self.tissue_decoder(torch.cat([tissue_separate_normal, tissue_shared_normal], dim=1))
        tissue_reconstructed_cancer = self.tissue_decoder(torch.cat([tissue_separate_cancer, tissue_shared_cancer], dim=1))

        # print('Done! output include: (share-dmso, share-normal, share-cancer), (recon-dmso, recon-trt), (recon-normal, recon-cancer), domain_preds')
        return (cellline_shared_dmso, tissue_shared_normal, tissue_shared_cancer), \
                (cellline_reconstructed_dmso, cellline_reconstructed_trt), \
                (tissue_reconstructed_normal, tissue_reconstructed_cancer), domain_preds

# Example initialization
# input_dim = 20
# hidden_dim = 64
# shared_dim = 1
# separate_dim = 5
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, hard=True):
        super(TripletLoss, self).__init__()
        self.margin = margin  # Margin value as defined in the triplet loss formula
        self.hard = hard  # Whether to use hard or soft triplet loss
        
    def euclidean_distance(self, embedding1, embedding2, batch_ptr):
        """
        Computes the pairwise Euclidean distance between two network embeddings.
        Each embedding represents multiple entities (e.g., genes) within a batch.
        
        Args:
        - embedding1: Tensor of shape (batch_size * n_genes, emb_dim)
        - embedding2: Tensor of shape (batch_size * n_genes, emb_dim)
        - batch_ptr: List of pointers indicating gene indices for each network in the batch.

        Returns:
        - network_distances: Tensor of shape (batch_size, batch_size), where
          each element (i, j) is the mean distance between network i (in embedding1)
          and network j (in embedding2).
        """
        batch_size = len(batch_ptr)-1
        n_genes = batch_ptr[1].item() if batch_size > 1 else len(embedding1)
        
        # Initialize pairwise distance matrix
        mean_distances = torch.zeros(batch_size, batch_size, device=embedding1.device)  # len(batch_ptr)是batch_size
        
        #"""
        # inefficient loops, but memory friendly 应该也慢不到哪去,可以都试试
        # Loop over networks in the batch to conpute average distances
        for i in range(batch_size):
            for j in range(batch_size):
                # Get the corresponoding sub-embeddings for networks i and j
                emb1 = embedding1[i*n_genes : (i+1)*n_genes]
                emb2 = embedding2[j*n_genes : (j+1)*n_genes]
                
                # Compute pairwise distances between genes in the two networks
                gene_distances = torch.norm(emb1 - emb2, p=2, dim=-1) # Shape: (n_genes, n_genes) torch.cdist(emb1, emb2) if all gene-pairs needed
                mean_distances[i,j] = gene_distances.mean()
                
        # print('\nMean distances distribution: %.2f (30 percent) ->  %.2f (30 percent) ->  %.2f (30 percent)' %
        #       (torch.quantile(mean_distances.detach(),0.3).item(), 
        #           torch.quantile(mean_distances.detach(),0.5).item(), 
        #               torch.quantile(mean_distances.detach(),0.7).item()))
        return mean_distances
        """ 
        
        # Compute the pairwise Euclidean distance between each gene (embedding1 and embedding2)
        # embedding1 and embedding2 are of shape (1000, 8)
        pair_emb = torch.cat([embedding1, embedding2])     #shape (batch_size*gene, emb)
        dot_product = torch.matmul(pair_emb, pair_emb.T) # shape (batch_size*gene, ~) 
        
        square_norm = torch.diag(dot_product) # squared L2 norm for each gene in each emb
        # We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).        

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = torch.maximum(distances, torch.tensor(0.0))
        
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0.0).float()
        distances = distances + mask * 1e-15

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)
        distances = distances[:len(embedding1), len(embedding1):]
        
        # Get a single scalar distance for each pair by averaging the distances across all genes (~1000)
        for i in range(batch_size):
            for j in range(batch_size):
                mean_distance = torch.diag(distances[i*n_genes:(i+1)*n_genes, j*n_genes:(j+1)*n_genes]).mean()
                mean_distances[i,j] = mean_distance
                # shape (batch_size, batch_size) row: anchor; col: pos/neg matrix = distance matrix of each emb pair
                # then choosing hard mode or random picking for triplet loss
    
        return mean_distances
        """
    
    def forward(self, anchor, positive, negative, batch_ptr):
        # Compute pairwise distances
        # shape:(batch_size * n_gene, emb)
        pos_distance = self.euclidean_distance(anchor, positive, batch_ptr)  # Euclidean distance (L2 norm)
        neg_distance = self.euclidean_distance(anchor, negative, batch_ptr)  # Euclidean distance (L2 norm)
        
        if self.hard:
            pos_distance = pos_distance.max(dim=1)[0]  # pos_hardest: max distance for each anchor
            neg_distance = neg_distance.min(dim=1)[0]  # neg_hardest: min distance for each anchor
        else:
            pos_distance = pos_distance.mean(dim=1)  # pos_average (centroid of anchor to batch-all postive)
            neg_distance = neg_distance.mean(dim=1)  # neg_average (centroid of anchor to batch-all negtive)
        
        # Compute triplet loss
        loss = torch.mean(torch.clamp(pos_distance - neg_distance + self.margin, min=0.0))  # Ensure loss is >= 0
        
        triplet_condition = (pos_distance + self.margin) < neg_distance # Condition to satisfy for correct triplet

        # Calculate accuracy: Number of correct triplets / Total number of triplets
        #accuracy = triplet_condition.float().mean().item()
        
        return loss, triplet_condition.detach() #accuracy


def train_model(model, dataloader_cell, dataloader_tissue, optimizer, criterion_recon, 
                criterion_adv, criterion_trip, alpha, device, lambda_recon=0.01):
    """
    Train the DrugRepurposingModel for one epoch.
    
    Args:
        model: DrugRepurposingModel instance.
        dataloader: DataLoader providing batches of cell-line and tissue data.
        optimizer: Optimizer for model parameters.
        criterion_recon: Loss function for reconstruction (e.g., MSELoss).
        criterion_adv: Loss function for adversarial (domain classification) (e.g., CrossEntropyLoss).
        alpha: Weight for the gradient reversal layer.
        device: Device to perform computations on (CPU or GPU).
        
    Returns:
        epoch_loss: Average loss for the epoch.
        domain_acc: Domain classification accuracy.
    """
    model.train()
    epoch_loss = 0
    all_preds, all_labels, all_tripAcc = [], [], []
    
    for batch_cell, batch_tissue in zip(dataloader_cell, dataloader_tissue):
        # Move batch data to device
        cellline_dmso, cellline_trt = batch_cell
        tissue_normal, tissue_cancer = batch_tissue
        domain_labels = torch.cat([tissue_cancer.t, tissue_normal.t, cellline_dmso.t, cellline_trt.t], dim=0)
        
        
        cellline_dmso = cellline_dmso.to(device); cellline_trt = cellline_trt.to(device)
        tissue_normal = tissue_normal.to(device); tissue_cancer = tissue_cancer.to(device)
        domain_labels = domain_labels.to(device)
        
        
        # Forward pass 
        (cellline_shared_dmso, tissue_shared_normal, tissue_shared_cancer), \
        (cellline_recon_dmso, cellline_recon_trt), (tissue_recon_normal, tissue_recon_cancer), domain_preds = \
            model(tissue_normal, tissue_cancer, cellline_trt, cellline_dmso, alpha)

        # Reconstruction Loss
        loss_recon_cellline_dmso = criterion_recon(cellline_recon_dmso, cellline_dmso.x)
        loss_recon_cellline_trt = criterion_recon(cellline_recon_trt, cellline_trt.x)
        loss_recon_tissue_normal = criterion_recon(tissue_recon_normal, tissue_normal.x)
        loss_recon_tissue_cancer = criterion_recon(tissue_recon_cancer, tissue_cancer.x)
        loss_recon = loss_recon_cellline_dmso + loss_recon_cellline_trt + loss_recon_tissue_normal + loss_recon_tissue_cancer

        # Adversarial Loss
        loss_adv = criterion_adv(domain_preds, domain_labels)
        
        # Triplet Loss
        loss_trip, acc_trip_condition = criterion_trip(anchor = tissue_shared_cancer, positive = cellline_shared_dmso, \
                                                       negative = tissue_shared_normal, batch_ptr = cellline_dmso.ptr)

        # Total Loss
        print('This train batch-> %.2f*loss_recon, loss_adv, loss_trip = %.2f, %.2f, %.2f' %(lambda_recon, lambda_recon*loss_recon, loss_adv, loss_trip))
        total_loss = lambda_recon*loss_recon + loss_adv + loss_trip
        


        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()                                                         

        # Accumulate metrics
        epoch_loss += total_loss.item()
        all_preds.append(torch.argmax(domain_preds, dim=1).detach().cpu())
        all_labels.append(domain_labels.detach().cpu())
        all_tripAcc.append(acc_trip_condition)

    # Calculate domain classification accuracy
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    domain_acc = accuracy_score(all_labels, all_preds)  # (all_labels==all_preds).sum()/len(all_labels)
    trip_acc = torch.cat(all_tripAcc).float().mean().item()

    return epoch_loss / len(dataloader_cell), domain_acc, trip_acc


def validate_model(model, dataloader_cell, dataloader_tissue, criterion_recon, 
                criterion_adv, criterion_trip, alpha, device, lambda_recon=0.01):
    """
    Validate the DrugRepurposingModel.
    
    Args:
        model: DrugRepurposingModel instance.
        dataloader: DataLoader providing validation batches.
        criterion_recon: Loss function for reconstruction (e.g., MSELoss).
        criterion_adv: Loss function for adversarial (domain classification) (e.g., CrossEntropyLoss).
        alpha: Weight for the gradient reversal layer.
        device: Device to perform computations on (CPU or GPU).
        
    Returns:
        val_loss: Average loss for validation.
        domain_acc: Domain classification accuracy.
    """
    model.eval()
    val_loss = 0
    all_preds, all_labels, all_tripAcc = [], [], []

    with torch.no_grad():
        for batch_cell, batch_tissue in zip(dataloader_cell, dataloader_tissue):
            # Move batch data to device
            cellline_dmso, cellline_trt = batch_cell
            tissue_normal, tissue_cancer = batch_tissue
            domain_labels = torch.cat([tissue_cancer.t, tissue_normal.t, cellline_dmso.t, cellline_trt.t], dim=0)
            
            # Move batch data to device
            cellline_dmso = cellline_dmso.to(device); cellline_trt = cellline_trt.to(device)
            tissue_normal = tissue_normal.to(device); tissue_cancer = tissue_cancer.to(device)
            domain_labels = domain_labels.to(device)
            

            # Forward pass
            (cellline_shared_dmso, tissue_shared_normal, tissue_shared_cancer), \
            (cellline_recon_dmso, cellline_recon_trt), (tissue_recon_normal, tissue_recon_cancer), domain_preds = \
                model(tissue_normal, tissue_cancer, cellline_trt, cellline_dmso, alpha)

            # Reconstruction Loss
            loss_recon_cellline_dmso = criterion_recon(cellline_recon_dmso, cellline_dmso.x)
            loss_recon_cellline_trt = criterion_recon(cellline_recon_trt, cellline_trt.x)
            loss_recon_tissue_normal = criterion_recon(tissue_recon_normal, tissue_normal.x)
            loss_recon_tissue_cancer = criterion_recon(tissue_recon_cancer, tissue_cancer.x)
            loss_recon = loss_recon_cellline_dmso + loss_recon_cellline_trt + loss_recon_tissue_normal + loss_recon_tissue_cancer

            # Adversarial Loss
            loss_adv = criterion_adv(domain_preds, domain_labels)
            
            # Triplet Loss
            loss_trip, acc_trip_condition= criterion_trip(anchor = tissue_shared_cancer, positive = cellline_shared_dmso, \
                                                          negative = tissue_shared_normal, batch_ptr = cellline_dmso.ptr)


            # Total Loss
            
            # Define your loss weights
            # lambda_recon = 0.005  # Example scaling factor for reconstruction loss
            # lambda_adv = 1.0      # No scaling for adversarial loss (or adjust if needed)
            # lambda_trip = 1.0     # No scaling for triplet loss (or adjust if needed)
            # total_loss = lambda_recon * loss_recon + lambda_adv * loss_adv + lambda_trip * loss_trip
            print('Val: %.2f*loss_recon, loss_adv, loss_trip = %.2f, %.2f, %.2f' %(lambda_recon, lambda_recon*loss_recon, loss_adv, loss_trip))
            total_loss = lambda_recon*loss_recon + loss_adv + loss_trip
            val_loss += total_loss.item()

            # Accumulate metrics
            all_preds.append(torch.argmax(domain_preds, dim=1).detach().cpu())
            all_labels.append(domain_labels.detach().cpu())
            all_tripAcc.append(acc_trip_condition)

    # Calculate domain classification accuracy
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    domain_acc = accuracy_score(all_labels, all_preds)
    trip_acc = torch.cat(all_tripAcc).float().mean().item()

    return val_loss / len(dataloader_cell), domain_acc, trip_acc                             



# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                torch.save(model.state_dict(), self.path.replace('.pth', '_final.pth'))
                print("Early stopping triggered.")
                return True
        return False

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

def pretrain_discriminator(model, train_cellloader, train_tissueloader, val_cellloader, val_tissueloader, 
                           epochs, device, lr=1e-3):
    """
    Pretrain only the discriminator for a few epochs.
    
    Args:
        model: The DrugRepurposingModel instance.
        train_loader: DataLoader containing the training data.
        discriminator_optimizer: Optimizer specifically for the discriminator.
        epochs: Number of pretraining epochs.
        device: Device to use (e.g., 'cuda' or 'cpu').
    
    Returns:
        None
    """
    model.train()  # Ensure the model is in training mode
    # Define a specific optimizer for the discriminator
    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(),  # Only optimize the discriminator's parameters
        lr=lr,  # Learning rate for discriminator
        weight_decay=1e-5  # Regularization
    )

    def pretrain(cellloader, tissueloader, train=True):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if train:
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        for batch_cell, batch_tissue in zip(cellloader, tissueloader):
            # domains: 0 for tissue, 1 for cell-line
            cellline_dmso, cellline_trt = batch_cell
            tissue_normal, tissue_cancer = batch_tissue
            domain_labels = torch.cat([tissue_cancer.t, tissue_normal.t, cellline_dmso.t, cellline_trt.t], dim=0) 

            # Move batch data to device
            cellline_dmso = cellline_dmso.to(device); cellline_trt = cellline_trt.to(device)
            tissue_normal = tissue_normal.to(device); tissue_cancer = tissue_cancer.to(device)
            domain_labels = domain_labels.to(device)
            
            # Forward pass for shared embeddings
            cellline_shared_trt = model.shared_encoder(cellline_trt).view(len(cellline_trt.t), cellline_trt.ptr[1], -1).mean(dim=1)
            cellline_shared_dmso = model.shared_encoder(cellline_dmso).view(len(cellline_dmso.t), cellline_dmso.ptr[1], -1).mean(dim=1)
            tissue_shared_normal = model.shared_encoder(tissue_normal).view(len(tissue_normal.t), tissue_normal.ptr[1], -1).mean(dim=1)
            tissue_shared_cancer = model.shared_encoder(tissue_cancer).view(len(tissue_cancer.t), tissue_cancer.ptr[1], -1).mean(dim=1)
            # Combine cell-line and tissue inputs
            shared_embedding = torch.cat([tissue_shared_cancer, tissue_shared_normal, cellline_shared_dmso, cellline_shared_trt], dim=0)  
            
            # Pass the embeddings to the discriminator
            domain_preds = model.discriminator(shared_embedding)
            
            # Compute the loss for the discriminator
            loss = F.cross_entropy(domain_preds, domain_labels)  # Cross-entropy loss for domain classification
            
            if train:
                # Backpropagation and optimization
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_correct += (domain_preds.argmax(dim=1) == domain_labels).sum().item()
            total_samples += domain_labels.size(0)
        
        # Logging for each epoch
        avg_loss = total_loss / len(cellloader)
        accuracy = total_correct / total_samples
        if train:
            print(f"Discriminator Pretraining-train Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}", end=' ')
        else:
            print(f"-val Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        model.train()
        for param in model.parameters():
            param.requires_grad = True

    for epoch in range(epochs):
        pretrain(train_cellloader, train_tissueloader, train=True)
        pretrain(val_cellloader, val_tissueloader, train=False)
    print("Discriminator pretraining complete.")

        
def training_pipeline(model, train_cellloader, train_tissueloader, val_cellloader, val_tissueloader, num_epochs, lr, device, 
                      pretrainD = False, pretrainD_epochs=5, lr_discriminator=1e-3, 
                      margin=0.2, lambda_recon=0.01, logfile='./Lr2_log.txt', warmup_epochs =3):
    """
    Train and validate the DrugRepurposingModel over multiple epochs.
    """
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8) 
    
    # Early stopping
    dir_mod = logfile.replace(logfile[logfile.find('log') : ], 'model_EstopEpoch.pth')
    early_stopping = EarlyStopping(patience=10, delta=0.001, path=dir_mod)

    
    # Loss functions
    criterion_recon = nn.MSELoss()       # For reconstruction
    criterion_adv = nn.CrossEntropyLoss()  # For domain classification. This expects raw logits (no activation) 集成了softmax操作,类似BCEWithLogitsLoss对sigmoid的集成
    criterion_trip = TripletLoss(margin=margin, hard=True)
    # Move model to device
    model.to(device)
    
    log = open(logfile, 'w')
    log.write('NOTION: lr_init = %f; drop/5ep: 0.8; nepochs = %d; n_batch = %d; margin = %.2f \n' % (lr, num_epochs, len(train_tissueloader), margin))
    log.write('NOTION: lambda_recon = %.3f; warmup_epochs = %d; pretrainD = %s; pretrainD_epochs = %d; lr_discriminator = %.4f \n' % (lambda_recon, warmup_epochs, pretrainD, pretrainD_epochs, lr_discriminator))
    log.write('NOTION: hidden_dim = 16; shared_dim = 16; separate_dim = 32; hidden_decoder_dim = 32; hidden_desc_dim = 16; nlayer_separate = 2; nlayer_shared = 2; nhead = 2; dp = 0.5 \n')
    

    for epoch in range(num_epochs):
        if pretrainD and epoch % 1000 == 0:   # 只在开始前预训练判别器
            pretrain_discriminator(model, train_cellloader, train_tissueloader, val_cellloader, val_tissueloader,
                                pretrainD_epochs, device, lr_discriminator)

        # p = float(epoch/num_epochs)
        # alpha = 2. / (1 + np.exp(-5 * p)) - 1 
        if epoch < warmup_epochs:
            alpha = 0.0
        else:
            p = float((epoch - warmup_epochs) / (num_epochs - warmup_epochs))
            alpha = 2. / (1 + np.exp(-5 * p)) - 1 
            
        # Train
        train_loss, train_domain_acc, train_trip_acc = train_model(
            model, train_cellloader, train_tissueloader, optimizer, criterion_recon, criterion_adv, criterion_trip, alpha, device, lambda_recon)

        # Validate
        val_loss, val_domain_acc, val_trip_acc = validate_model(
            model, val_cellloader, val_tissueloader, criterion_recon, criterion_adv,  criterion_trip, alpha, device, lambda_recon)

        print(f"\nEpoch {epoch+1}/{num_epochs} -> LR = {optimizer.param_groups[0]['lr']}")
        print(f"Train Loss: {train_loss:.4f}, Train Domain Acc: {train_domain_acc:.4f}, Train Trip Acc: {train_trip_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Domain Acc: {val_domain_acc:.4f}, Val Trip Acc: {val_trip_acc:.4f}")
        print('============================================================\n')
        scheduler_lr.step()
        
        
        
        log.write(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f};  Train Domain Acc: {train_domain_acc:.4f}; \
                  Train Trip Acc: {train_trip_acc:.4f}; Val Loss: {val_loss:.4f};  Val Domain Acc: {val_domain_acc:.4f}; \
                            Val Trip Acc: {val_trip_acc:.4f}; LR: {optimizer.param_groups[0]["lr"]}; GRL-alpha: {alpha:.2f}\n')
        

        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"{name} is unused in the forward pass.")
                
        # Check early stopping     
        fake_loss = 1 - val_trip_acc 
        if early_stopping(val_loss, model):   # early_stopping(fake_loss, model):    # early_stopping(val_loss, model)
            print(f"Training stopped early at epoch {epoch+1}")
            break

    
    print('Run Finished. Log file saved as: %s; Final model diction saved as: %s' % (logfile, dir_mod))
    log.close()
    # torch.save(model, '{0}/model_epoch_current_{1}.pth'.format(model_root, fname))
    # torch.save(model.state_dict(), dir_mod)
    
    # Load best model (early stopping will save the best model state)
    # model.load_state_dict(torch.load('best_model.pth'))
    # return model

# 查看模型隐层
def viewEmb(model, dataloader_cell, dataloader_tissue, domain_out = False):
    device = model.parameters().__next__().device
    model.eval()
    
    emb_cell_dmso, emb_cell_trt, emb_tissue_normal, emb_tissue_cancer = [], [], [], []
    domain_preds = []

    with torch.no_grad():
        for batch_cell, batch_tissue in zip(dataloader_cell, dataloader_tissue):
            # Move batch data to device
            cellline_dmso, cellline_trt = batch_cell
            tissue_normal, tissue_cancer = batch_tissue

            # Move batch data to device
            cellline_dmso = cellline_dmso.to(device); cellline_trt = cellline_trt.to(device)
            tissue_normal = tissue_normal.to(device); tissue_cancer = tissue_cancer.to(device)
            
            cellline_shared_trt = model.shared_encoder(cellline_trt).view(len(cellline_trt.t), cellline_trt.ptr[1], -1).mean(dim=1)
            cellline_shared_dmso = model.shared_encoder(cellline_dmso).view(len(cellline_dmso.t), cellline_dmso.ptr[1], -1).mean(dim=1)
            tissue_shared_normal = model.shared_encoder(tissue_normal).view(len(tissue_normal.t), tissue_normal.ptr[1], -1).mean(dim=1)
            tissue_shared_cancer = model.shared_encoder(tissue_cancer).view(len(tissue_cancer.t), tissue_cancer.ptr[1], -1).mean(dim=1)
            emb_cell_dmso.append(cellline_shared_dmso)
            emb_cell_trt.append(cellline_shared_trt)
            emb_tissue_normal.append(tissue_shared_normal)
            emb_tissue_cancer.append(tissue_shared_cancer)

            if domain_out:
                domain_pred = model.discriminator(torch.cat([tissue_shared_cancer, tissue_shared_normal, cellline_shared_dmso, cellline_shared_trt], dim=0))
                domain_preds.append(domain_pred)
    emb_cell_dmso = torch.cat(emb_cell_dmso).cpu().numpy()
    emb_cell_trt = torch.cat(emb_cell_trt).cpu().numpy()
    emb_tissue_normal = torch.cat(emb_tissue_normal).cpu().numpy()
    emb_tissue_cancer = torch.cat(emb_tissue_cancer).cpu().numpy()
    emb = pd.DataFrame(np.concatenate([emb_tissue_cancer, emb_tissue_normal, emb_cell_dmso, emb_cell_trt], axis=0))
    emb.index =  ['cancer']*len(emb_tissue_cancer) +  ['normal']*len(emb_tissue_normal) + \
        ['dmso']*len(emb_cell_dmso) + ['trt']*len(emb_cell_trt)
    if domain_out:
        domain_preds = torch.cat(domain_preds).cpu().numpy()
        domain_preds = np.argmax(domain_preds, axis=1)
        return emb, domain_preds
    return emb


# Fine-tuning
def fineTune(model, pth, train_cellloader, train_tissueloader, val_cellloader, val_tissueloader, num_epochs, lr, device, 
                      pretrainD = False, pretrainD_epochs=5, lr_discriminator=1e-3, 
                      margin=0.2, lambda_recon=0.001, logfile='./FT_log.txt', warmup_epochs =0):
    
    model.load_state_dict(torch.load(pth, map_location=torch.device(device)))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8) 
    
    # Early stopping
    dir_mod = logfile.replace(logfile[logfile.find('log') : ], 'model_FtEpoch.pth')
    early_stopping = EarlyStopping(patience=10, delta=0.001, path=dir_mod)
    
    # Loss functions
    criterion_recon = nn.MSELoss()       # For reconstruction
    criterion_adv = nn.CrossEntropyLoss()  # For domain classification. This expects raw logits (no activation) 集成了softmax操作,类似BCEWithLogitsLoss对sigmoid的集成
    criterion_trip = TripletLoss(margin=margin, hard=True)
    # Move model to device
    model.to(device)

    log = open(logfile, 'w')
    log.write('FineTuning NOTION: lr_init = %f; drop/5ep: 0.8; nepochs = %d; n_batch = %d; margin = %.2f \n' % (lr, num_epochs, len(train_tissueloader), margin))
    log.write('FineTuning NOTION: lambda_recon = %.3f; warmup_epochs = %d; pretrainD = %s; pretrainD_epochs = %d; lr_discriminator = %.4f \n' % (lambda_recon, warmup_epochs, pretrainD, pretrainD_epochs, lr_discriminator))
    log.write('FineTuning NOTION: hidden_dim = 16; shared_dim = 16; separate_dim = 32; hidden_decoder_dim = 32; hidden_desc_dim = 16; nlayer_separate = 2; nlayer_shared = 2; nhead = 2; dp = 0.5 \n')
    
    for epoch in range(num_epochs):
        # train discriminator alone
        if pretrainD and epoch % 1000 == 0:  # Update discriminator every 2 epochs
            pretrain_discriminator(model, train_cellloader, train_tissueloader, val_cellloader, val_tissueloader, 
                                   pretrainD_epochs, device, lr_discriminator)

        if epoch < warmup_epochs:
            alpha = 0.0
        else:
            p = float((epoch - warmup_epochs) / (num_epochs - warmup_epochs))
            alpha = 2. / (1 + np.exp(-5 * p)) - 1 
            
        # Train model 
        train_loss, train_domain_acc, train_trip_acc = train_model(
            model, train_cellloader, train_tissueloader, optimizer, criterion_recon, criterion_adv, criterion_trip, alpha, device, lambda_recon)

        # Validate
        val_loss, val_domain_acc, val_trip_acc = validate_model(
            model, val_cellloader, val_tissueloader, criterion_recon, criterion_adv,  criterion_trip, alpha, device, lambda_recon)

        print(f"\nEpoch {epoch+1}/{num_epochs} -> LR = {optimizer.param_groups[0]['lr']}")
        print(f"Train Loss: {train_loss:.4f}, Train Domain Acc: {train_domain_acc:.4f}, Train Trip Acc: {train_trip_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Domain Acc: {val_domain_acc:.4f}, Val Trip Acc: {val_trip_acc:.4f}")
        print('============================================================\n')
        scheduler_lr.step()        
        
        log.write(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f};  Train Domain Acc: {train_domain_acc:.4f}; \
                  Train Trip Acc: {train_trip_acc:.4f}; Val Loss: {val_loss:.4f};  Val Domain Acc: {val_domain_acc:.4f}; \
                            Val Trip Acc: {val_trip_acc:.4f}; LR: {optimizer.param_groups[0]["lr"]}; GRL-alpha: {alpha:.2f}\n')
        
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"{name} is unused in the forward pass.")
                
        # Check early stopping  
        fake_loss = 1 - val_trip_acc   # 1-trip_acc as loss for early stopping
        if early_stopping(val_loss, model):   #early_stopping(fake_loss, model):
            print(f"Training stopped early at epoch {epoch+1}")
            break

    print('Run Fine-tuning Finished. Log file saved as: %s; Final model diction saved as: %s' % (logfile, dir_mod))
    log.close()

# trt按照zscore，不做分位数标准化
from sklearn.preprocessing import StandardScaler
def prepare_and_zscore(df2, dataset):
    # df: gene*sample
    if df2.shape[1] >= 20:
        h24 = [k for k in df2.columns if '_24H_' in k]
        hoth = [k for k in df2.columns if '_24H_' not in k]
        if len(h24) >= 20:
            df2 = df2.loc[:, h24].sample(20, axis=1)
        else:
            noth = 20-len(h24)
            df2 = pd.concat([df2.loc[:, h24], df2.loc[:, hoth].sample(noth, axis=1)], axis=1)
    else:
        random.seed(1234)
        n = 20 - df2.shape[1]; groups = []
        while len(groups) < n :
            g = random.sample(range(df2.shape[1]), 2)
            if all(len(set(g).intersection(ig)) < 2 for ig in groups):
                groups.append(g)
        for i in groups:
            j = df2.iloc[:, i[0]] * 0.5 + df2.iloc[:, i[1]] * 0.5
            col_i = df2.shape[1]
            df2 = pd.concat([df2, j], axis=1)
    # 标准化
    scaler = StandardScaler()
    right = scaler.fit_transform(df2.T) # sample*feature
    edge = dataset.makeGraph(pd.DataFrame(right)) # 输入是sample*gene
    data = Data(x = torch.tensor(right.T, dtype=torch.float32), edge_index=edge, t=1) # t=1 表示cell-line
    return data


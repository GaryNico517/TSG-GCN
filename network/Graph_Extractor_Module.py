import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv,BatchNorm
import scipy.sparse as sp
import pandas as pd
import networkx as nx
import warnings
from network import gcn,graph
from module.DAML import AM_Generate_Pred

warnings.filterwarnings('ignore')

Adjacency_Matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


class GBA(nn.Module):
    def __init__(self, input_num,output_channels,hidden_layer, class_num = 10):
        super(GBA, self).__init__()
        self.class_num = class_num

        # self.conv3x3 = nn.Conv3d(in_channels=input_num, out_channels=hidden_layer, kernel_size=3, padding=1, bias=True)
        # self.Norm3x3 = nn.BatchNorm3d(num_features=hidden_layer)
        # self.relu3x3 = nn.ReLU(inplace=True)

        self.featuremap_2_graph = gcn.My_Featuremaps_to_Graph(input_channels=input_num, hidden_layers=hidden_layer,
                                                           nodes=self.class_num)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layer, hidden_layer)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layer, hidden_layer)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layer, hidden_layer)

        self.graph_2_fea = gcn.My_Graph_to_Featuremaps(hidden_layers=hidden_layer, output_channels=output_channels)

    def get_graph(self):
        spine_adj = graph.preprocess_adj(graph.tooth_graph)
        spine_adj_ = torch.from_numpy(spine_adj).float()
        spine_adj = spine_adj_.unsqueeze(0).cuda()
        return spine_adj

    def get_graph_from_MMIP(self,MMIP):
        # print(MMIP.shape,MMIP.requires_grad)
        adj_from_MMIP = AM_Generate_Pred(MMIP)
        # print(adj_from_MMIP,adj_from_MMIP.requires_grad)
        # spine_adj = graph.normalize_adj_torch(torch.tensor(Adjacency_Matrix.astype(np.float32)) + torch.eye(Adjacency_Matrix.shape[0]))
        spine_adj = graph.preprocess_adj_torch(adj_from_MMIP)
        spine_adj = spine_adj.cuda()
        return spine_adj

    def forward(self, x, MMIP = None):
        # x = self.conv3x3(x)
        # x = self.Norm3x3(x)
        # x = self.relu3x3(x)
        if MMIP is None:
            adj = self.get_graph()
        else:
            adj = self.get_graph_from_MMIP(MMIP)
        spine_graph,fea_logit = self.featuremap_2_graph(x)

        # graph convolution by 3 times
        spine_graph = self.graph_conv1.forward(spine_graph, adj=adj, relu=True)
        spine_graph = self.graph_conv2.forward(spine_graph, adj=adj, relu=True)
        spine_graph = self.graph_conv3.forward(spine_graph, adj=adj, relu=True)
        # graph -> feature map
        spine_graph = self.graph_2_fea.forward(spine_graph, fea_logit)

        return spine_graph,fea_logit
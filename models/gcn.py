import numpy as np
from dgl.nn.pytorch import GraphConv
import dgl
import torch


# class GCNGraphNew(torch.nn.Module):
#     def __init__(self, in_feats, h_feats):
#         super(GCNGraphNew, self).__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, h_feats)
#         self.conv3 = GraphConv(h_feats, h_feats)
#         self.dense = torch.nn.Linear(h_feats, 1)
#         self.maxpool = dgl.nn.pytorch.glob.MaxPooling()

#     def forward(self, g, in_feat, e_weight):
#         h = self.conv1(g, in_feat, e_weight)
#         h = torch.nn.functional.relu(h)
#         h = self.conv2(g, h, e_weight)
#         h = torch.nn.functional.relu(h)
#         h = self.conv3(g, h, e_weight)
#         h = torch.nn.functional.relu(h)
#         g.ndata['h'] = h
#         h = self.maxpool(g, h)  # pooling
#         h = self.dense(h)
#         h = torch.nn.functional.sigmoid(h)
#         return h

class GCNGraph(torch.nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCNGraph, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, h_feats)
        self.dense1 = torch.nn.Linear(h_feats, 16)
        self.dense2 = torch.nn.Linear(16, 8)
        self.dense3 = torch.nn.Linear(8, 1)

    def forward(self, g, in_feat, e_weight):
        h = self.conv1(g, in_feat, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv3(g, h, e_weight)
        h = torch.nn.functional.relu(h)
        g.ndata['h'] = h
        h = dgl.mean_nodes(g, 'h')  # pooling
        h = self.dense1(h)
        h = torch.nn.functional.relu(h)
        h = self.dense2(h)
        h = torch.nn.functional.relu(h)
        h = self.dense3(h)
        h = torch.nn.functional.sigmoid(h)
        return h


class GCNNodeBAShapes(torch.nn.Module):
    # TODO
    def __init__(self, in_feats, h_feats, num_classes, device, if_exp=False):
        super(GCNNodeBAShapes, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, num_classes)
        self.if_exp = if_exp
        self.device = device

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat((torch.tensor([0]).to(self.device), torch.cumsum(g.batch_num_nodes(), dim=0)), dim=0)[:-1]
        target_node = target_node + x
        h = self.conv1(g, in_feat, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv3(g, h, e_weight)
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h = torch.nn.functional.softmax(h)
        g.ndata['h'] = h
        return g.ndata['h'][target_node]


class GCNNodeTreeCycles(torch.nn.Module):
    # TODO
    def __init__(self, in_feats, h_feats, num_classes, if_exp=False):
        super(GCNNodeTreeCycles, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, num_classes)
        self.if_exp = if_exp

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat((torch.tensor([0]), torch.cumsum(g.batch_num_nodes(), dim=0)), dim=0)[:-1]
        target_node = target_node + x

        h = self.conv1(g, in_feat, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv3(g, h, e_weight)
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h = torch.nn.functional.sigmoid(h)
        g.ndata['h'] = h
        return g.ndata['h'][target_node]


class GCNNodeCiteSeer(torch.nn.Module):
    # TODO
    def __init__(self, in_feats, h_feats, num_classes, if_exp=False):
        super(GCNNodeCiteSeer, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.if_exp = if_exp

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat((torch.tensor([0]), torch.cumsum(g.batch_num_nodes(), dim=0)), dim=0)[:-1]
        target_node = target_node + x

        h = self.conv1(g, in_feat, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h, e_weight)
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h = torch.nn.functional.softmax(h)
        g.ndata['h'] = h
        return g.ndata['h'][target_node]

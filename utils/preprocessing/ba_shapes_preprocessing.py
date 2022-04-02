"""Read the Mutag dataset and create the graphx"""

import numpy as np
import os
import dgl
from dgl.data import DGLDataset
import torch
import networkx as nx
import matplotlib.pyplot as plt
from dgl import save_graphs, load_graphs
from utils.common_utils import read_file
from utils.common_utils import ba_shapes_dgl_to_networkx


class BAShapesDataset(DGLDataset):
    def __init__(self, adj=None, node_labels=None, edge_labels=None, hop_num=3, feat_dim=10, load_path=None):
        super().__init__(name='ba_shapes')
        if load_path:
            self.load_path = load_path
            self.load_()
        else:
            self.adj = adj
            self.edge_labels = edge_labels
            self.node_labels = node_labels
            self.hop_num = hop_num
            self.feat_dim = feat_dim
            self.graphs = []
            self.labels = []
            self.targets = []

            for n_i, node in enumerate(np.arange(len(self.adj))):
                n_l = self.node_labels[node]
                g, new_idx = self.sub_graph_generator(node)
                self.graphs.append(g)
                self.labels.append(n_l)
                self.targets.append(new_idx)
            self.labels = torch.from_numpy(np.array(self.labels))
            self.targets = torch.from_numpy(np.array(self.targets))

    def sub_graph_generator(self, node):
        """
        a simple bfs to find the k-hop sub-graph
        :param node:
        :param node_labels:
        :return:
        """
        sub_nodes = set()  # the sub nodes in the sub graph (within k hop)
        sub_nodes.add(node)
        que = [node]
        close_set = set()
        for i in range(self.hop_num):
            hop_nodes = []
            while que:
                tar = que.pop(0)
                neighbors = np.where(self.adj[tar] == 1)[0]
                hop_nodes.extend(neighbors)
                sub_nodes.update(neighbors)
                if tar not in close_set:
                    close_set.add(tar)
            if len(hop_nodes) == 0:
                break
            for n in hop_nodes:
                if n not in close_set:
                    que.append(n)
        sub_nodes = np.sort(np.array(list(sub_nodes)))
        node_new = np.where(sub_nodes == node)[0][0]
        sub_edge_labels = self.edge_labels[sub_nodes][:, sub_nodes]
        filtered_sub_edge_labels = np.zeros((sub_edge_labels.shape[0], sub_edge_labels.shape[1]))

        sgt_nodes = set()  # the sub nodes in the gt graph (within k hop)
        sgt_nodes.add(node_new)
        que = [node_new]
        close_set = set()
        for i in range(self.hop_num + 1):
            hop_nodes = []
            while que:
                tar = que.pop(0)
                neighbors = np.where(sub_edge_labels[tar] == 1)[0]
                hop_nodes.extend(neighbors)
                for n in neighbors:
                    filtered_sub_edge_labels[tar, n] = 1
                    filtered_sub_edge_labels[n, tar] = 1
                sgt_nodes.update(neighbors)
                if tar not in close_set:
                    close_set.add(tar)
            if len(hop_nodes) == 0:
                break
            for n in hop_nodes:
                if n not in close_set:
                    que.append(n)
        sub_edge_labels = filtered_sub_edge_labels
        sub_adj = self.adj[sub_nodes][:, sub_nodes]
        sub_nodes = np.arange(len(sub_nodes))

        # create dgl graph
        comb = np.array(np.meshgrid(sub_nodes, sub_nodes)).T.reshape(-1, 2)
        g = dgl.graph((torch.from_numpy(comb[:, 0]), torch.from_numpy(comb[:, 1])), num_nodes=len(sub_nodes))
        g_feats = np.ones((len(sub_nodes), self.feat_dim))
        g.ndata['feat'] = torch.from_numpy(g_feats)
        edge_weights = sub_adj.reshape(1, -1)[0]
        edge_gts = sub_edge_labels.reshape(1, -1)[0]
        g.edata['weight'] = torch.from_numpy(edge_weights)
        g.edata['gt'] = torch.from_numpy(edge_gts)
        return g, node_new

    def process(self):
        print('processing')

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i], self.targets[i]

    def __len__(self):
        return len(self.graphs)

    def save_(self, save_path):
        save_graphs(os.path.join(save_path, 'dgl_graph.bin'), self.graphs, {'labels': self.labels})
        np.array(self.targets).dump(os.path.join(save_path, 'targets.pickle'))

    def load_(self):
        # load processed data from directory `self.save_path`
        self.graphs, label_dict = load_graphs(os.path.join(self.load_path, 'dgl_graph.bin'))
        self.labels = label_dict['labels']
        self.feat_dim = self.graphs[0].ndata['feat'].shape[1]
        self.targets = np.load(os.path.join(self.load_path, 'targets.pickle'), allow_pickle=True)


def ba_shapes_preprocessing(dataset_dir):
    name = "BA_Shapes"
    data = np.load(os.path.join(dataset_dir, 'syn_data.pkl'), allow_pickle=True)
    adj = np.array(data[0], dtype='float32')
    feats = data[1]
    y_train = data[2]
    y_val = data[3]
    y_test = data[4]
    e_labels = data[8]
    e_labels = np.array(np.maximum(e_labels, e_labels.T), dtype="float32")  # make symmetric
    node_labels = np.array(np.logical_or(y_train, np.logical_or(y_val, y_test)), dtype=int)
    G_dataset = BAShapesDataset(adj, node_labels, e_labels, hop_num=3, feat_dim=10)
    return G_dataset

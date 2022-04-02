"""Read the Mutag dataset and create the graphx"""

import numpy as np
import os
import dgl
from dgl.data import DGLDataset
import torch
import networkx as nx
import matplotlib.pyplot as plt
from dgl import save_graphs, load_graphs
from utils.common_utils import read_file_citeseer
from utils.common_utils import ba_shapes_dgl_to_networkx


class CiteSeerDataset(DGLDataset):
    def __init__(self, adj=None, node_labels=None, node_feats=None, hop_num=3, load_path=None):
        super().__init__(name='citeseer')
        if load_path:
            self.load_path = load_path
            self.load_()
        else:
            self.adj = adj
            self.node_feats = node_feats
            self.node_labels = node_labels
            self.hop_num = hop_num
            self.feat_dim = len(node_feats[0])
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
        # print(node)
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

        sub_adj = self.adj[sub_nodes][:, sub_nodes]
        g_feats = self.node_feats[sub_nodes]

        sub_nodes = np.arange(len(sub_nodes))
        # create dgl graph
        comb = np.array(np.meshgrid(sub_nodes, sub_nodes)).T.reshape(-1, 2)
        g = dgl.graph((torch.from_numpy(comb[:, 0]), torch.from_numpy(comb[:, 1])), num_nodes=len(sub_nodes))
        g.ndata['feat'] = torch.from_numpy(g_feats)
        edge_weights = sub_adj.reshape(1, -1)[0]
        g.edata['weight'] = torch.from_numpy(edge_weights)
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


def citeseer_preprocessing(dataset_dir):
    name = "citeseer"
    paper_type_dict = {"Agents": 0, "AI": 1, "DB": 2, "IR": 3, "ML": 4, "HCI":  5}
    edge_data_path = os.path.join(dataset_dir, 'citeseer.cites')
    node_info_data_path = os.path.join(dataset_dir, 'citeseer.content')
    node_info_data = np.array(read_file_citeseer(node_info_data_path))

    edge_data = np.array(read_file_citeseer(edge_data_path))

    # filter out papers without info

    valid_paper_set = set()
    for info in node_info_data:
        valid_paper_set.add(info[0])

    valid_edge_data = []
    for edge in edge_data:
        if edge[0] in valid_paper_set and edge[1] in valid_paper_set:
            valid_edge_data.append(edge)
    edge_data = np.array(valid_edge_data)  # only the edges with info

    name_int_dict = {}  # {'name': index}
    idx = 0
    for edge in edge_data:
        if edge[0] not in name_int_dict:
            name_int_dict[edge[0]] = idx
            idx += 1
        if edge[1] not in name_int_dict:
            name_int_dict[edge[1]] = idx
            idx += 1

    for i in range(len(edge_data)):
        edge_data[i][0] = name_int_dict[edge_data[i][0]]
        edge_data[i][1] = name_int_dict[edge_data[i][1]]

    edge_data = np.array(edge_data, dtype=int)

    node_num = len(name_int_dict.keys())
    feat_dim = len(node_info_data[0][1:-1])
    node_labels = np.ones(node_num, dtype=int) * -1

    node_feats = np.ones((node_num, feat_dim)) * -1
    idx_set = set()
    for i in range(len(node_info_data)):
        paper_id = node_info_data[i][0]
        paper_label = paper_type_dict[node_info_data[i][-1]]
        paper_feat = node_info_data[i][1:-1]
        paper_idx = name_int_dict[paper_id]
        idx_set.add(paper_idx)
        node_labels[paper_idx] = paper_label
        node_feats[paper_idx] = paper_feat

    # create adj matrix
    adj = np.zeros((node_num, node_num), dtype='float32')
    for edge in edge_data:
        n0 = edge[0]
        n1 = edge[1]
        adj[n0, n1] = 1
        adj[n1, n0] = 1

    G_dataset = CiteSeerDataset(adj, node_labels, node_feats, hop_num=3)
    return G_dataset

"""Read the Mutag dataset and create the graphx"""

import numpy as np
import os
import dgl
from dgl.data import DGLDataset
import torch
from dgl import save_graphs, load_graphs
from utils.common_utils import read_file


class MutagDataset0(DGLDataset):
    def __init__(self, edges=None, graph_indicator=None, node_labels=None, edge_labels=None, graph_labels=None, load_path=None):
        super().__init__(name='mutag0')
        if load_path:
            self.load_path = load_path
            self.load_()
        else:
            for i in range(len(graph_labels)):
                if graph_labels[i] == 1:
                    graph_labels[i] = 0
                else:
                    graph_labels[i] = 1

            self.edges = edges
            self.graph_indicator = graph_indicator
            self.node_labels = node_labels
            self.edge_labels = edge_labels
            self.graph_labels = graph_labels

            self.graphs = []
            self.labels = []
            self.feat_dim = len(np.unique(self.node_labels))
            self.component_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                      8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
            # group edges
            edges_group = {}
            e_labels_group = {}
            for e_id, edge in enumerate(self.edges):
                e_label = self.edge_labels[e_id]
                g_id = self.graph_indicator[edge[0]]
                if g_id != self.graph_indicator[edge[1]]:
                    print('graph indicator error!')
                    exit(1)
                if g_id not in edges_group.keys():
                    edges_group[g_id] = [edge]
                    e_labels_group[g_id] = [e_label]
                else:
                    edges_group[g_id].append(edge)
                    e_labels_group[g_id].append(e_label)

            for g_id, g_edges in edges_group.items():
                g_label = self.graph_labels[g_id]
                g_edges = np.array(g_edges)
                g_e_labels = e_labels_group[g_id]
                src = g_edges[:, 0]
                dst = g_edges[:, 1]
                unique_nodes = np.unique(np.concatenate((src, dst), axis=0))
                g_feats = np.zeros((len(unique_nodes), self.feat_dim))
                int_feats = self.node_labels[unique_nodes]
                g_feats[np.arange(len(unique_nodes)), int_feats] = 1  # convert feature to one-hot vec
                n_id_dict = {}
                n_id_dict_reverse = {}
                for i in range(len(unique_nodes)):
                    n_id_dict[unique_nodes[i]] = i
                    n_id_dict_reverse[i] = unique_nodes[i]
                for i in range(len(src)):
                    src[i] = n_id_dict[src[i]]
                    dst[i] = n_id_dict[dst[i]]
                num_nodes = len(np.unique(np.concatenate((src, dst), axis=0)))
                adj = np.zeros((num_nodes, num_nodes), dtype='float32')
                adj_e_label = np.zeros((num_nodes, num_nodes), dtype='float32')
                for i in range(len(src)):
                    n0 = src[i]
                    n1 = dst[i]
                    adj[n0, n1] = 1.0
                    adj_e_label[n0, n1] = g_e_labels[i]
                comb = np.array(np.meshgrid(np.arange(num_nodes), np.arange(num_nodes))).T.reshape(-1, 2)
                g = dgl.graph((torch.from_numpy(comb[:, 0]), torch.from_numpy(comb[:, 1])), num_nodes=num_nodes)
                g.ndata['feat'] = torch.from_numpy(g_feats)
                edge_weights = adj.reshape(1, -1)[0]
                edge_labels = adj_e_label.reshape(1, -1)[0]
                g.edata['weight'] = torch.from_numpy(edge_weights)
                g.edata['label'] = torch.from_numpy(edge_labels)

                self.graphs.append(g)
                self.labels.append(g_label)
            self.labels = torch.from_numpy(np.array(self.labels))

    def process(self):
        print('processing')

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def save_(self, save_path):
        save_graphs(save_path, self.graphs, {'labels': self.labels})

    def load_(self):
        # load processed data from directory `self.save_path`
        self.graphs, label_dict = load_graphs(os.path.join(self.load_path, 'dgl_graph.bin'))
        self.labels = label_dict['labels']
        self.feat_dim = self.graphs[0].ndata['feat'].shape[1]


def mutag_preprocessing_0(dataset_dir):
    name = "Mutagenicity"
    edge_path = os.path.join(dataset_dir, name + "_A.txt")
    edge_label_path = os.path.join(dataset_dir, name + "_edge_gt.txt")
    graph_indicator_path = os.path.join(dataset_dir, name + "_graph_indicator.txt")
    node_label_path = os.path.join(dataset_dir, name + "_node_labels.txt")
    graph_label_path = os.path.join(dataset_dir, name + "_graph_labels.txt")
    edge_data = read_file(edge_path)
    edge_data = np.array(edge_data)
    edge_data = edge_data - 1
    edge_labels = read_file(edge_label_path)
    edge_labels = np.array(edge_labels)
    graph_indicator = read_file(graph_indicator_path) - 1
    node_labels = read_file(node_label_path)
    graph_labels = read_file((graph_label_path))
    G_dataset = MutagDataset0(edge_data, graph_indicator, node_labels, edge_labels, graph_labels)
    return G_dataset

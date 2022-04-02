import numpy as np
import networkx as nx


def mutag_dgl_to_networkx(dgl_G):
    component_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                      8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
    nodes = dgl_G.nodes().numpy()
    edges = np.array(list(zip(dgl_G.edges()[0], dgl_G.edges()[1])))
    node_labels = dgl_G.ndata['feat'].numpy()
    edge_weights = dgl_G.edata['weight'].numpy()
    edge_labels = dgl_G.edata['label'].numpy()
    edges = edges[np.where(edge_weights > 0)]
    edge_labels = edge_labels[np.where(edge_weights > 0)]
    nx_G = nx.Graph()
    nx_G.add_nodes_from(nodes)
    # add edge with label
    for eid in range(len(edges)):
        nx_G.add_edge(edges[eid][0], edges[eid][1], gt=edge_labels[eid])
    for node in nx_G.nodes(data=True):
        node[1]['label'] = component_dict[np.where(node_labels[node[0]] == 1.0)[0][0]]
    return nx_G


def get_mutag_color_dict():
    mutage_color_dict = {'C': 'tab:orange', 'O': 'tab:gray', 'Cl': 'cyan', 'H': 'tab:blue', 'N': 'blue',
                       'F': 'green', 'Br': 'y', 'S': 'm', 'P': 'red', 'I': 'tab:green', 'Na': 'tab: purple',
                       'K': 'tab:brown', 'Li': 'tab:pink', 'Ca': 'tab:olive'}
    return mutage_color_dict


def read_file(f_path):
    """
    read graph dataset .txt files
    :param f_path: the path to the .txt file
    :return: read the file (as lines) and return numpy arrays.
    """
    f_list = []
    with open(f_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.replace('\n', '').split(',')
            f_list.append([])
            for item in items:
                f_list[-1].append(int(item))
    return np.array(f_list).squeeze()


def read_file_citeseer(f_path):
    """
    read citeseer dataset
    :param f_path: the path to the .txt file
    :return: read the file (as lines) and return numpy arrays.
    """
    f_list = []
    with open(f_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.replace('\n', '').split('\t')
            f_list.append([])
            for item in items:
                f_list[-1].append(item)
    return np.array(f_list).squeeze()


def ba_shapes_dgl_to_networkx(dgl_G):
    nodes = dgl_G.nodes().numpy()
    edges = np.array(list(zip(dgl_G.edges()[0], dgl_G.edges()[1])))
    edge_weights = dgl_G.edata['weight'].numpy()
    edge_labels = dgl_G.edata['gt'].numpy()
    edges = edges[np.where(edge_weights > 0)]
    edge_labels = edge_labels[np.where(edge_weights > 0)]
    nx_G = nx.Graph()  # init networkX graph
    nx_G.add_nodes_from(nodes)
    # add edge with label
    for eid in range(len(edges)):
        nx_G.add_edge(edges[eid][0], edges[eid][1], gt=edge_labels[eid])
    return nx_G


def citeseer_dgl_to_networkx(dgl_G):
    nodes = dgl_G.nodes().numpy()
    edges = np.array(list(zip(dgl_G.edges()[0], dgl_G.edges()[1])))
    edge_weights = dgl_G.edata['weight'].numpy()
    edge_labels = dgl_G.edata['gt'].numpy()
    edges = edges[np.where(edge_weights > 0)]
    edge_labels = edge_labels[np.where(edge_weights > 0)]
    nx_G = nx.Graph()  # init networkX graph
    nx_G.add_nodes_from(nodes)
    # add edge with label
    for eid in range(len(edges)):
        nx_G.add_edge(edges[eid][0], edges[eid][1], gt=edge_labels[eid])
    return nx_G
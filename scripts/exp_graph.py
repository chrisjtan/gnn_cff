import os
import numpy as np
import torch
from utils.argument import arg_parse_exp_graph_mutag_0
from models.explainer_models import GraphExplainerEdge
from models.gcn import GCNGraph
from utils.preprocessing.mutag_preprocessing_0 import MutagDataset0
import sys


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    torch.manual_seed(0)
    np.random.seed(0)
    exp_args = arg_parse_exp_graph_mutag_0()
    print("argument:\n", exp_args)
    model_path = exp_args.model_path
    train_indices = np.load(os.path.join(model_path, 'train_indices.pickle'), allow_pickle=True)
    test_indices = np.load(os.path.join(model_path, 'test_indices.pickle'), allow_pickle=True)
    G_dataset = MutagDataset0(load_path=os.path.join(model_path))
    graphs = G_dataset.graphs
    labels = G_dataset.labels
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    base_model = GCNGraph(G_dataset.feat_dim, 128).to(device)
    base_model.load_state_dict(torch.load(os.path.join(model_path, 'model.model')))
    #  fix the base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Create explainer
    explainer = GraphExplainerEdge(
        base_model=base_model,
        G_dataset=G_dataset,
        args=exp_args,
        test_indices=test_indices,
        # fix_exp=15
    )

    explainer.explain_nodes_gnn_stats()

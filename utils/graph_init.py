import numpy as np


def graph_init_real(dataset):
    if dataset == "Mutagenicity_0":
        from utils.preprocessing.mutag_preprocessing_0 import mutag_preprocessing_0
        G_dataset = mutag_preprocessing_0(dataset_dir="datasets/Mutagenicity_0")
    elif dataset == "NCI1":
        from utils.preprocessing.nci1_preprocessing import nci1_preprocessing
        G_dataset = nci1_preprocessing(dataset_dir="datasets/NCI1")
    elif dataset == "BA_Shapes":
        from utils.preprocessing.ba_shapes_preprocessing import ba_shapes_preprocessing
        G_dataset = ba_shapes_preprocessing(dataset_dir="datasets/BA_Shapes")
    elif dataset == "Tree_Cycles":
        from utils.preprocessing.tree_cycles_preprocessing import tree_cycles_preprocessing
        G_dataset = tree_cycles_preprocessing(dataset_dir="datasets/Tree_Cycles")
    elif dataset == "citeseer":
        from utils.preprocessing.citeseer_preprocessing import citeseer_preprocessing
        G_dataset = citeseer_preprocessing(dataset_dir="datasets/citeseer")
    else:
        print('Error: Dataset not known.')
        exit(2)
    return G_dataset
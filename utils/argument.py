import argparse


def arg_parse_train_graph_mutag_0():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="Mutagenicity_0",
                        help="choose a graph classification task")
    parser.add_argument("--gpu", dest="gpu", action="store_false", help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default='0.005', help="L2 norm to the wights")
    parser.add_argument("--opt", dest="opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--train_ratio", dest="train_ratio", type=float, default=0.8, help="ratio of training data")
    parser.add_argument("--lr", dest="lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1000, help="number of the training epochs")
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="log")
    return parser.parse_args()


def arg_parse_train_node_ba_shapes():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="BA_Shapes",
                        help="choose a node classification task")
    parser.add_argument("--gpu", dest="gpu", action="store_false", help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default='0.005', help="L2 norm to the wights")
    parser.add_argument("--opt", dest="opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--train_ratio", dest="train_ratio", type=float, default=0.8, help="ratio of training data")
    parser.add_argument("--lr", dest="lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=3000, help="number of the training epochs")
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="log")
    return parser.parse_args()


def arg_parse_train_node_tree_cycles():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="Tree_Cycles",
                        help="choose a node classification task")
    parser.add_argument("--gpu", dest="gpu", action="store_false", help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default='0.005', help="L2 norm to the wights")
    parser.add_argument("--opt", dest="opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--train_ratio", dest="train_ratio", type=float, default=0.8, help="ratio of training data")
    parser.add_argument("--lr", dest="lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1000, help="number of the training epochs")
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="log")
    return parser.parse_args()


def arg_parse_exp_graph_mutag_0():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="Mutagenicity_0",
                        help="choose a graph explanation task")
    parser.add_argument("--model_path", dest="model_path", type=str, default="log/Mutagenicity_0_logs",
                        help="path to the model that need to be explained")
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default='0.005', help="L2 norm to the wights")
    parser.add_argument("--opt", dest="opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", dest="lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=500, help="number of the training epochs")
    parser.add_argument("--lam", dest="lam", type=float, default=1000,
                        help="hyper param control the trade-off between "
                             "the explanation complexity and explanation strength")
    parser.add_argument("--alp", dest="alp", type=float, default=0.6,
                        help="hyper param control factual and counterfactual, 1 is totally factual")
    parser.add_argument("--gam", dest="gam", type=float, default=.5, help="margin value for bpr loss")
    parser.add_argument("--mask_thresh", dest="mask_thresh", type=float, default=.5,
                        help="threshold to convert relaxed adj matrix to binary")
    return parser.parse_args()


def arg_parse_exp_node_ba_shapes():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="BA_Shapes",
                        help="choose a node explanation task")
    parser.add_argument("--model_path", dest="model_path", type=str, default="log/BA_Shapes_logs",
                        help="path to the model that need to be explained")
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default='0.005', help="L2 norm to the wights")
    parser.add_argument("--opt", dest="opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", dest="lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=2000, help="number of the training epochs")
    parser.add_argument("--lam", dest="lam", type=float, default=500,
                        help="hyper param control the trade-off between "
                             "the explanation complexity and explanation strength")
    parser.add_argument("--alp", dest="alp", type=float, default=0.6,
                        help="hyper param control factual and counterfactual")
    parser.add_argument("--gam", dest="gam", type=float, default=0.5, help="margin value for bpr loss")
    parser.add_argument("--mask_thresh", dest="mask_thresh", type=float, default=.5,
                        help="threshold to convert relaxed adj matrix to binary")
    return parser.parse_args()


def arg_parse_exp_node_tree_cycles():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="Tree_Cycles",
                        help="choose a node explanation task")
    parser.add_argument("--model_path", dest="model_path", type=str, default="log/Tree_Cycles_logs",
                        help="path to the model that need to be explained")
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default='0.005', help="L2 norm to the wights")
    parser.add_argument("--opt", dest="opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", dest="lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=500, help="number of the training epochs")
    parser.add_argument("--lam", dest="lam", type=float, default=500,
                        help="hyper param control the trade-off between "
                             "the explanation complexity and explanation strength")
    parser.add_argument("--alp", dest="alp", type=float, default=0.6,
                        help="hyper param control factual and counterfactual")
    parser.add_argument("--gam", dest="gam", type=float, default=0.5, help="margin value for bpr loss")
    parser.add_argument("--mask_thresh", dest="mask_thresh", type=float, default=.1,
                        help="threshold to convert relaxed adj matrix to binary")
    return parser.parse_args()
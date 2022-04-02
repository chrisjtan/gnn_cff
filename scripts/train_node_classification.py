import numpy as np
import torch
import os
import time
from pathlib import Path
from models.gcn import GCNNodeBAShapes
from utils.argument import arg_parse_train_node_ba_shapes
from utils.graph_init import graph_init_real
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader


def train_node_classification(args):
    if args.gpu:
        device = torch.device('cuda:%s' % args.cuda)
    else:
        device = 'cpu'
    # device = 'cpu'
    out_path = os.path.join(args.save_dir, args.dataset + "_logs")
    G_dataset = graph_init_real(args.dataset)

    Path(out_path).mkdir(parents=True, exist_ok=True)
    num_examples = len(G_dataset)

    num_train = int(num_examples * args.train_ratio)
    train_indices = np.unique(np.random.choice(np.arange(num_examples), num_train, replace=False))
    test_indices = np.unique(np.array([i for i in np.arange(num_examples) if i not in train_indices]))

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = GraphDataLoader(
        G_dataset, sampler=train_sampler, batch_size=32, drop_last=False)
    test_dataloader = GraphDataLoader(
        G_dataset, sampler=test_sampler, batch_size=32, drop_last=False)

    model = GCNNodeBAShapes(G_dataset.feat_dim, 16, num_classes=4, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    begin = time.time()

    for epoch in range(args.num_epochs):
        losses = []
        num_correct = 0
        num_train = 0
        # begin = time.time()
        for batched_graph, labels, target_nodes in train_dataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            target_nodes = target_nodes.to(device)
            optimizer.zero_grad()
            pred = model(batched_graph, batched_graph.ndata['feat'].float(),
                         batched_graph.edata['weight'], target_nodes).squeeze()
            # print(pred)
            ori_int_labels = torch.argmax(labels, dim=1)
            pre_int_labels = torch.argmax(pred, dim=1)
            num_correct += (ori_int_labels == pre_int_labels).sum().item()
            num_train += len(labels)
            loss = loss_fn(pred, ori_int_labels)
            losses.append(loss.to('cpu').detach().numpy())
            loss.backward()
            optimizer.step()
        print('epoch:%d' % epoch, 'loss:',  np.mean(losses), 'Train accuracy:', num_correct / num_train)

    # evaluate
    num_correct = 0
    num_train = 0
    for batched_graph, labels, target_nodes in train_dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        target_nodes = target_nodes.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat'].float(),
                     batched_graph.edata['weight'], target_nodes).squeeze()
        ori_int_labels = torch.argmax(labels, dim=1)
        pre_int_labels = torch.argmax(pred, dim=1)
        num_correct += (ori_int_labels == pre_int_labels).sum().item()
        num_train += len(labels)
    print('Final train accuracy:', num_correct / num_train)

    num_correct = 0
    num_tests = 0
    for batched_graph, labels, target_nodes in test_dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        target_nodes = target_nodes.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat'].float(),
                     batched_graph.edata['weight'], target_nodes).squeeze()
        ori_int_labels = torch.argmax(labels, dim=1)
        pre_int_labels = torch.argmax(pred, dim=1)
        num_correct += (ori_int_labels == pre_int_labels).sum().item()
        num_tests += len(labels)
    print('Test accuracy:', num_correct / num_tests)

    print('time: ', time.time() - begin)
    train_indices.dump(os.path.join(out_path, 'train_indices.pickle'))
    test_indices.dump(os.path.join(out_path, 'test_indices.pickle'))
    G_dataset.save_(out_path)
    torch.save(model.state_dict(), os.path.join(out_path, 'model.model'))
    return True


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    train_args = arg_parse_train_node_ba_shapes()
    if train_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = train_args.cuda
        print("Using CUDA", train_args.cuda)
    else:
        print("Using CPU")
    print(train_args)

    train_node_classification(train_args)

import numpy as np
import torch
import os
import time
from pathlib import Path
from models.gcn import GCNGraph
from utils.argument import arg_parse_train_graph_mutag_0
from utils.graph_init import graph_init_real
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader


def train_graph_classification(args):
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
        G_dataset, sampler=train_sampler, batch_size=128, drop_last=False)
    test_dataloader = GraphDataLoader(
        G_dataset, sampler=test_sampler, batch_size=128, drop_last=False)
    model = GCNGraph(G_dataset.feat_dim, 128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(args.num_epochs):
        begin = time.time()
        losses = []
        num_correct = 0
        num_train = 0
        for batched_graph, labels in train_dataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(batched_graph, batched_graph.ndata['feat'].float(), batched_graph.edata['weight']).squeeze()
            num_correct += ((pred >= 0.5).int() == labels).sum().item()
            num_train += len(labels)
            loss = loss_fn(pred, labels.float())
            losses.append(loss.to('cpu').detach().numpy())
            loss.backward()
            optimizer.step()
        print('epoch:%d' % epoch, 'loss:',  np.mean(losses), 'Train accuracy:', num_correct / num_train)

        print('time', time.time() - begin)
    # evaluate
    num_correct = 0
    num_tests = 0
    for batched_graph, labels in train_dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat'].float(), batched_graph.edata['weight']).squeeze()
        num_correct += ((pred >= 0.5).int() == labels).sum().item()
        num_tests += len(labels)
    print('Final train accuracy:', num_correct / num_tests)
    num_correct = 0
    num_tests = 0
    for batched_graph, labels in test_dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat'].float(), batched_graph.edata['weight']).squeeze()
        num_correct += ((pred >= 0.5).int() == labels).sum().item()
        num_tests += len(labels)
    print('Test accuracy:', num_correct / num_tests)
    train_indices.dump(os.path.join(out_path, 'train_indices.pickle'))
    test_indices.dump(os.path.join(out_path, 'test_indices.pickle'))
    G_dataset.save_(os.path.join(out_path, 'dgl_graph.bin'))
    torch.save(model.state_dict(), os.path.join(out_path, 'model.model'))
    return True


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    train_args = arg_parse_train_graph_mutag_0()
    if train_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = train_args.cuda
        print("Using CUDA", train_args.cuda)
    else:
        print("Using CPU")
    train_graph_classification(train_args)

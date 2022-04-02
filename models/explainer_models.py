from re import S
import numpy as np
import torch
import math
import tqdm
import sys
import matplotlib.pyplot as plt
import networkx as nx
from utils.common_utils import mutag_dgl_to_networkx, get_mutag_color_dict, ba_shapes_dgl_to_networkx


class GraphExplainerEdge(torch.nn.Module):
    def __init__(self, base_model, G_dataset, test_indices, args, fix_exp=None):
        super(GraphExplainerEdge, self).__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.G_dataset = G_dataset
        self.test_indices = test_indices
        self.args = args
        if fix_exp:
            self.fix_exp = fix_exp * 2
        else:
            self.fix_exp = None

    def explain_nodes_gnn_stats(self):
        exp_dict = {}  # {'gid': masked_adj, 'gid': mask_adj}
        num_dict = {}  # {'gid': exp_num, 'gid': exp_num}
        num=200
        for gid in tqdm.tqdm(self.test_indices[:num]):
            ori_pred = self.base_model(self.G_dataset.graphs[gid],
                                       self.G_dataset.graphs[gid].ndata['feat'].float(),
                                       self.G_dataset.graphs[gid].edata['weight'])[0, 0]
            pred_label = torch.round(ori_pred)
            ori_label = self.G_dataset.labels[gid]
            if pred_label == 1 and ori_label == 1:  # only explain why the graph is predicted as mutagenic
                masked_adj, exp_num = self.explain(gid, ori_pred)
                exp_dict[gid] = masked_adj
                num_dict[gid] = exp_num
        print('average number of exps:', sum(num_dict.values()) / len(num_dict.keys()))
        PN = self.compute_pn(exp_dict)
        PS = self.compute_ps(exp_dict)
        acc, pre, rec, f1 = self.compute_precision_recall(exp_dict)
        print('PN', PN)
        print('PS', PS)
        print('FNS', 2 * PN * PS / (PN + PS))
        print('acc: ', acc, ' pre: ', pre, ' rec: ', rec, ' f1: ', f1)
        return PN, PS, 2 * PN * PS / (PN + PS), sum(num_dict.values()) / len(num_dict.keys()), acc, pre, rec, f1

    def explain(self, gid, ori_pred):
        # only generate exps for the correct predictions for now (to be consistent with GNN Explainer).
        explainer = ExplainModelGraph(
            graph=self.G_dataset.graphs[gid],
            base_model=self.base_model,
            args=self.args
        )
        if self.args.gpu:
            explainer = explainer.cuda()
        # train explainer
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.args.lr, weight_decay=0)
        explainer.train()
        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            pred1, pred2 = explainer()

            bpr1, bpr2, l1, loss = explainer.loss(
                pred1[0, 0], pred2[0, 0], ori_pred, self.args.gam, self.args.lam, self.args.alp)

            # if epoch % 200 == 0:
            #     print('bpr1: ', self.args.lam * self.args.alp * bpr1,
            #           'bpr2:', self.args.lam * (1 - self.args.alp) * bpr2,
            #           'l1', l1,
            #           'loss', loss)
                # print('bpr: ', 50 * bpr, 'l1', l1, 'loss', loss)
            loss.backward()
            optimizer.step()

        masked_adj = explainer.get_masked_adj()
        masked_adj = explainer.get_masked_adj()
        new_edge_num = len(masked_adj[masked_adj > self.args.mask_thresh])
        exp_num = new_edge_num / 2
        return masked_adj, exp_num

    def compute_pn(self, exp_dict):
        pn_count = 0
        for gid, masked_adj in exp_dict.items():
            graph = self.G_dataset.graphs[gid]
            if self.fix_exp:
                thresh = masked_adj.flatten().sort(descending=True)[0][self.fix_exp+1]
            else:
                thresh = self.args.mask_thresh
            ps_adj = (masked_adj > thresh).float()
            pn_adj = graph.edata['weight'] - ps_adj
            new_pre = self.base_model(graph, graph.ndata['feat'].float(), pn_adj)[0, 0]
            if new_pre < 0.5:
                pn_count += 1
        pn = pn_count / len(exp_dict.keys())
        return pn

    def compute_ps(self, exp_dict):
        ps_count = 0
        for gid, masked_adj in exp_dict.items():
            graph = self.G_dataset.graphs[gid]
            if self.fix_exp:
                thresh = masked_adj.flatten().sort(descending=True)[0][self.fix_exp+1]
            else:
                thresh = self.args.mask_thresh
            ps_adj = (masked_adj > thresh).float()
            new_pre = self.base_model(graph, graph.ndata['feat'].float(), ps_adj)[0, 0]
            if new_pre > 0.5:
                ps_count += 1
        ps = ps_count / len(exp_dict.keys())
        return ps

    def compute_precision_recall(self, exp_dict):
        pres = []
        recalls = []
        f1s = []
        accs = []
        for gid, masked_adj in exp_dict.items():
            if self.fix_exp:
                thresh = masked_adj.flatten().sort(descending=True)[0][self.fix_exp+1]
            else:
                thresh = self.args.mask_thresh
            e_labels = self.G_dataset[gid][0].edata['label']
            new_edges = [masked_adj > thresh][0].numpy()
            old_edges = [self.G_dataset[gid][0].edata['weight'] > thresh][0].numpy()
            int_map = map(int, new_edges)
            new_edges = list(int_map)
            int_map = map(int, old_edges)
            old_edges = list(int_map)

            exp_list = np.array(new_edges)
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i in range(len(e_labels)):
                if exp_list[i] == 1:
                    if e_labels[i] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if e_labels[i] == 1:
                        FN += 1
                    else:
                        TN += 1
            if TP != 0:
                pre = TP / (TP + FP)
                rec = TP / (TP + FN)
                acc = (TP + TN) / (TP + FP + TN + FN)
                f1 = 2 * pre * rec / (pre + rec)
            else:
                pre = 0
                rec = 0
                f1 = 0
                acc = (TP + TN) / (TP + FP + TN + FN)
            pres.append(pre)
            recalls.append(rec)
            f1s.append(f1)
            accs.append(acc)
        return np.mean(accs), np.mean(pres), np.mean(recalls), np.mean(f1s)


class ExplainModelGraph(torch.nn.Module):
    def __init__(self, graph, base_model, args):
        super(ExplainModelGraph, self).__init__()
        self.graph = graph
        self.num_nodes = len(self.graph.nodes())
        self.base_model = base_model
        self.args = args
        self.adj_mask = self.construct_adj_mask()
        # For masking diagonal entries
        self.diag_mask = torch.ones(self.num_nodes, self.num_nodes) - torch.eye(self.num_nodes)
        if self.args.gpu:
            self.diag_mask = self.diag_mask.cuda()

    def forward(self):
        masked_adj = self.get_masked_adj()
        # should be reversed in the future
        pred1 = self.base_model(self.graph, self.graph.ndata['feat'].float(), masked_adj)  # factual
        pred2 = self.base_model(self.graph, self.graph.ndata['feat'].float(), self.graph.edata['weight'] - masked_adj)  # counterfactual
        return pred1, pred2

    def loss(self, pred1, pred2, ori_pred, gam, lam, alp):
        relu = torch.nn.ReLU()
        bpr1 = relu(gam + 0.5 - pred1)  # factual
        bpr2 = relu(gam + pred2 - 0.5)  # counterfactual
        masked_adj = self.get_masked_adj()
        L1 = torch.linalg.norm(masked_adj, ord=1)
        loss = L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
        return bpr1, bpr2, L1, loss

    def construct_adj_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.num_nodes + self.num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def get_masked_adj(self):
        sym_mask = torch.sigmoid(self.adj_mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.graph.edata['weight']
        flatten_sym_mask = torch.reshape(sym_mask, (-1, ))
        masked_adj = adj * flatten_sym_mask
        # masked_adj = masked_adj * self.diag_mask
        ''
        return masked_adj


class NodeExplainerEdgeMulti(torch.nn.Module):
    def __init__(self, base_model, G_dataset, test_indices, args, fix_exp=None):
        super(NodeExplainerEdgeMulti, self).__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.G_dataset = G_dataset
        self.test_indices = test_indices
        self.args = args
        if fix_exp:
            self.fix_exp = fix_exp * 2
        else:
            self.fix_exp = None

    def explain_nodes_gnn_stats(self):
        exp_dict = {}  # {'gid': masked_adj, 'gid': mask_adj}
        num_dict = {}  # {'gid': exp_num, 'gid': exp_num}
        pred_label_dict = {}
        t_gid = []
        for gid in tqdm.tqdm(self.test_indices):
            ori_pred = self.base_model(self.G_dataset.graphs[gid],
                                       self.G_dataset.graphs[gid].ndata['feat'].float(),
                                       self.G_dataset.graphs[gid].edata['weight'], self.G_dataset.targets[gid])[0]
            ori_pred_label = torch.argmax(ori_pred)
            if self.args.dataset == 'citeseer':
                ori_label = self.G_dataset.labels[gid]
            else:
                ori_label = torch.argmax(self.G_dataset.labels[gid])
            if self.args.dataset == 'citeseer' or (ori_pred_label != 0 and ori_label != 0):
                t_gid.append(gid)
                masked_adj, exp_num = self.explain(gid, ori_pred_label)
                exp_dict[gid] = masked_adj
                num_dict[gid] = exp_num
                pred_label_dict[gid] = ori_pred_label
        print('average number of exps:', sum(num_dict.values()) / len(num_dict.keys()))

        PN = self.compute_pn(exp_dict, pred_label_dict)
        PS = self.compute_ps(exp_dict, pred_label_dict)
        if self.args.dataset == 'citeseer':
            acc = -1
            pre = -1
            rec = -1
            f1 = -1
        else:
            acc, pre, rec, f1 = self.compute_precision_recall(exp_dict)
        print('PN', PN)
        print('PS', PS)
        print('PNS', 2 * PN * PS / (PN + PS))
        print('ave exp', sum(num_dict.values()) / len(num_dict.keys()))
        print('acc: ', acc, ' pre: ', pre, ' rec: ', rec, ' f1: ', f1)
        return PN, PS, 2 * PN * PS / (PN + PS), sum(num_dict.values()) / len(num_dict.keys()), acc, pre, rec, f1

    def explain(self, gid, pred_label):
        explainer = ExplainModelNodeMulti(
            graph=self.G_dataset.graphs[gid],
            base_model=self.base_model,
            target_node=self.G_dataset.targets[gid],
            args=self.args
        )
        if self.args.gpu:
            explainer = explainer.cuda()
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.args.lr, weight_decay=0)
        explainer.train()
        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            pred1, pred2 = explainer()
            bpr1, bpr2, l1, loss = explainer.loss(
                pred1[0], pred2[0], pred_label, self.args.gam, self.args.lam, self.args.alp)

            # if epoch % 201 == 0:
            #     print('bpr1: ', self.args.lam * self.args.alp * bpr1,
            #           'bpr2:', self.args.lam * (1 - self.args.alp) * bpr2,
            #           'l1', l1,
            #           'loss', loss)
            loss.backward()
            optimizer.step()
        
        masked_adj = explainer.get_masked_adj()
        new_edge_num = len(masked_adj[masked_adj > self.args.mask_thresh])
        exp_num = new_edge_num / 2
        return masked_adj, exp_num

    def compute_pn(self, exp_dict, pred_label_dict):
        pn_count = 0
        for gid, masked_adj in exp_dict.items():
            graph = self.G_dataset.graphs[gid]
            target = self.G_dataset.targets[gid]
            ori_pred_label = pred_label_dict[gid]
            if self.fix_exp:
                if self.fix_exp > (len(masked_adj.flatten()) - 1):
                    thresh = masked_adj.flatten().sort(descending=True)[0][len(masked_adj.flatten()) - 1]
                else:
                    thresh = masked_adj.flatten().sort(descending=True)[0][self.fix_exp + 1]
            else:
                thresh = self.args.mask_thresh
            ps_adj = (masked_adj > thresh).float()
            pn_adj = graph.edata['weight'] - ps_adj
            new_pre = self.base_model(graph, graph.ndata['feat'].float(), pn_adj, target)[0]
            new_label = torch.argmax(new_pre)
            if new_label != ori_pred_label:
                pn_count += 1
        pn = pn_count / len(exp_dict.keys())
        return pn

    def compute_ps(self, exp_dict, pred_label_dict):
        ps_count = 0
        for gid, masked_adj in exp_dict.items():
            graph = self.G_dataset.graphs[gid]
            target = self.G_dataset.targets[gid]
            ori_pred_label = pred_label_dict[gid]
            if self.fix_exp:
                if self.fix_exp > (len(masked_adj.flatten()) - 1):
                    thresh = masked_adj.flatten().sort(descending=True)[0][len(masked_adj.flatten()) - 1]
                else:
                    thresh = masked_adj.flatten().sort(descending=True)[0][self.fix_exp + 1]
            else:
                thresh = self.args.mask_thresh
            ps_adj = (masked_adj > thresh).float()
            new_pre = self.base_model(graph, graph.ndata['feat'].float(), ps_adj, target)[0]
            new_label = torch.argmax(new_pre)
            if new_label == ori_pred_label:
                ps_count += 1
        ps = ps_count / len(exp_dict.keys())
        return ps

    def compute_precision_recall(self, exp_dict):
        pres = []
        recalls = []
        f1s = []
        accs = []

        for gid, masked_adj in exp_dict.items():
            if self.fix_exp:
                if self.fix_exp > (len(masked_adj.flatten()) - 1):
                    thresh = masked_adj.flatten().sort(descending=True)[0][len(masked_adj.flatten()) - 1]
                else:
                    thresh = masked_adj.flatten().sort(descending=True)[0][self.fix_exp + 1]
            else:
                thresh = self.args.mask_thresh
            e_labels = self.G_dataset[gid][0].edata['gt']
            new_edges = [masked_adj > thresh][0].numpy()
            old_edges = [self.G_dataset[gid][0].edata['weight'] > thresh][0].numpy()
            int_map = map(int, new_edges)
            new_edges = list(int_map)
            int_map = map(int, old_edges)
            old_edges = list(int_map)
            exp_list = np.array(new_edges)
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i in range(len(e_labels)):
                if exp_list[i] == 1:
                    if e_labels[i] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if e_labels[i] == 1:
                        FN += 1
                    else:
                        TN += 1
            # print('TP', TP, 'FP', FP, 'TN', TN, 'FN', FN)
            if TP != 0:
                pre = TP / (TP + FP)
                rec = TP / (TP + FN)
                acc = (TP + TN) / (TP + FP + TN + FN)
                f1 = 2 * pre * rec / (pre + rec)
            else:
                pre = 0
                rec = 0
                f1 = 0
                acc = (TP + TN) / (TP + FP + TN + FN)
            pres.append(pre)
            recalls.append(rec)
            f1s.append(f1)
            accs.append(acc)
        return np.mean(accs), np.mean(pres), np.mean(recalls), np.mean(f1s)

                
class ExplainModelNodeMulti(torch.nn.Module):
    """
    explain BA-shapes and CiteSeer
    """
    def __init__(self, graph, base_model, target_node, args):
        super(ExplainModelNodeMulti, self).__init__()
        self.graph = graph
        self.num_nodes = len(self.graph.nodes())
        self.base_model = base_model
        self.target_node = target_node
        self.args = args
        self.adj_mask = self.construct_adj_mask()
        # For masking diagonal entries
        self.diag_mask = torch.ones(self.num_nodes, self.num_nodes) - torch.eye(self.num_nodes)
        if self.args.gpu:
            self.diag_mask = self.diag_mask.cuda()

    def forward(self):
        masked_adj = self.get_masked_adj()
        pred1 = self.base_model(self.graph, self.graph.ndata['feat'].float(),
                                masked_adj, self.target_node)
        pred2 = self.base_model(self.graph, self.graph.ndata['feat'].float(),
                                self.graph.edata['weight'] - masked_adj,
                                self.target_node)
        return pred1, pred2

    def loss(self, pred1, pred2, pred_label, gam, lam, alp):
        relu = torch.nn.ReLU()
        f_next = torch.max(torch.cat((pred1[:pred_label],
                                      pred1[pred_label+1:])))
        cf_next = torch.max(torch.cat((pred2[:pred_label],
                                       pred2[pred_label+1:])))
        bpr1 = relu(gam + f_next - pred1[pred_label])
        bpr2 = relu(gam + pred2[pred_label] - cf_next)
        masked_adj = self.get_masked_adj()
        L1 = torch.linalg.norm(masked_adj, ord=1)
        loss = L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
        return bpr1, bpr2, L1, loss

    def construct_adj_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.num_nodes + self.num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def get_masked_adj(self):
        sym_mask = torch.sigmoid(self.adj_mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.graph.edata['weight']
        flatten_sym_mask = torch.reshape(sym_mask, (-1, ))
        masked_adj = adj * flatten_sym_mask
        ''
        return masked_adj


class NodeExplainerFeatureMulti(torch.nn.Module):
    def __init__(self, base_model, G_dataset, test_indices, args, fix_exp=None):
        super(NodeExplainerFeatureMulti, self).__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.G_dataset = G_dataset
        self.test_indices = test_indices
        self.args = args
        if fix_exp:
            self.fix_exp = fix_exp * 2
        else:
            self.fix_exp = None

    def explain_nodes_gnn_stats(self):
        exp_dict = {}  # {'gid': masked_adj, 'gid': mask_adj}
        num_dict = {}  # {'gid': exp_num, 'gid': exp_num}
        pred_label_dict = {}

        for gid in tqdm.tqdm(self.test_indices[:51]):
            ori_pred = self.base_model(self.G_dataset.graphs[gid],
                                       self.G_dataset.graphs[gid].ndata['feat'].float(),
                                       self.G_dataset.graphs[gid].edata['weight'], self.G_dataset.targets[gid])[0]
            ori_pred_label = torch.argmax(ori_pred)
            if self.args.dataset == 'citeseer':
                ori_label = self.G_dataset.labels[gid]
            else:
                ori_label = torch.argmax(self.G_dataset.labels[gid])
            if self.args.dataset == 'citeseer' or (ori_pred_label != 0 and ori_label != 0):  # only explain when the graph is not on the motif
                print('explain gid: ', gid)
                print('num of nodes: ', torch.sum(self.G_dataset[gid][0].edata['weight']))
                masked_feat, exp_num = self.explain(gid, ori_pred_label)
                exp_dict[gid] = masked_feat
                num_dict[gid] = exp_num
                pred_label_dict[gid] = ori_pred_label
        print('average number of exps:', sum(num_dict.values()) / len(num_dict.keys()))
        PN = self.compute_pn(exp_dict, pred_label_dict)
        PS = self.compute_ps(exp_dict, pred_label_dict)
        if self.args.dataset == 'citeseer':
            acc = -1
            pre = -1
            rec = -1
            f1 = -1
        else:
            acc, pre, rec, f1 = self.compute_precision_recall(exp_dict)
        print('PN', PN)
        print('PS', PS)
        print('PNS', 2 * PN * PS / (PN + PS))
        print('ave exp', sum(num_dict.values()) / len(num_dict.keys()))
        print('acc: ', acc, ' pre: ', pre, ' rec: ', rec, ' f1: ', f1)
        return PN, PS, 2 * PN * PS / (PN + PS), sum(num_dict.values()) / len(num_dict.keys()), acc, pre, rec, f1

    def explain(self, gid, pred_label):
        # only generate exps for the correct predictions for now (to be consistent with GNN Explainer).
        explainer = ExplainModelNodeMultiFeature(
            graph=self.G_dataset.graphs[gid],
            base_model=self.base_model,
            target_node=self.G_dataset.targets[gid],
            args=self.args
        )
        print('ori label', self.G_dataset.labels[gid])
        print('ori feat num', torch.sum(self.G_dataset.graphs[gid].ndata['feat']))

        if self.args.gpu:
            explainer = explainer.cuda()
        # train explainer
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.args.lr, weight_decay=0)
        explainer.train()
        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            pred1, pred2 = explainer()
            bpr1, bpr2, l1, loss = explainer.loss(
                pred1[0], pred2[0], pred_label, self.args.gam, self.args.lam, self.args.alp)

            if epoch % 200 == 0:
                print('bpr1: ', self.args.lam * self.args.alp * bpr1,
                      'bpr2:', self.args.lam * (1 - self.args.alp) * bpr2,
                      'l1', l1,
                      'loss', loss)

            loss.backward()
            optimizer.step()

        masked_feat = explainer.get_masked_feat()
        new_feat_num = len(masked_feat[masked_feat > self.args.mask_thresh])
        exp_num = new_feat_num
        print('exp num', exp_num)
        return masked_feat, exp_num

    def compute_pn(self, exp_dict, pred_label_dict):
        pn_count = 0
        for gid, masked_feat in exp_dict.items():
            graph = self.G_dataset.graphs[gid]
            target = self.G_dataset.targets[gid]
            ori_pred_label = pred_label_dict[gid]
            if self.fix_exp:
                thresh = masked_feat.flatten().sort(descending=True)[0][self.fix_exp+1]
            else:
                thresh = self.args.mask_thresh
            ps_feat = (masked_feat > thresh).float()
            pn_feat = graph.ndata['feat'] - ps_feat
            new_pre = self.base_model(graph, pn_feat.float(), graph.edata['weight'], target)[0]
            new_label = torch.argmax(new_pre)
            if new_label != ori_pred_label:
                pn_count += 1
        pn = pn_count / len(exp_dict.keys())
        return pn

    def compute_ps(self, exp_dict, pred_label_dict):
        ps_count = 0
        for gid, masked_feat in exp_dict.items():
            graph = self.G_dataset.graphs[gid]
            target = self.G_dataset.targets[gid]
            ori_pred_label = pred_label_dict[gid]
            if self.fix_exp:
                thresh = masked_feat.flatten().sort(descending=True)[0][self.fix_exp+1]
            else:
                thresh = self.args.mask_thresh
            ps_feat = (masked_feat > thresh).float()
            new_pre = self.base_model(graph, ps_feat.float(), graph.edata['weight'], target)[0]
            new_label = torch.argmax(new_pre)
            if new_label == ori_pred_label:
                ps_count += 1
        ps = ps_count / len(exp_dict.keys())
        return ps

    def compute_precision_recall(self, exp_dict):
        pres = []
        recalls = []
        f1s = []
        accs = []
        for gid, masked_adj in exp_dict.items():
            if self.fix_exp:
                thresh = masked_adj.flatten().sort(descending=True)[0][self.fix_exp+1]
            else:
                thresh = self.args.mask_thresh
            e_labels = self.G_dataset[gid][0].edata['gt']
            new_edges = [masked_adj > thresh][0].numpy()
            old_edges = [self.G_dataset[gid][0].edata['weight'] > thresh][0].numpy()
            int_map = map(int, new_edges)
            new_edges = list(int_map)
            int_map = map(int, old_edges)
            old_edges = list(int_map)

            exp_list = np.array(new_edges)
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i in range(len(e_labels)):
                if exp_list[i] == 1:
                    if e_labels[i] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if e_labels[i] == 1:
                        FN += 1
                    else:
                        TN += 1
            if TP != 0:
                pre = TP / (TP + FP)
                rec = TP / (TP + FN)
                acc = (TP + TN) / (TP + FP + TN + FN)
                f1 = 2 * pre * rec / (pre + rec)
            else:
                pre = 0
                rec = 0
                f1 = 0
                acc = (TP + TN) / (TP + FP + TN + FN)
            pres.append(pre)
            recalls.append(rec)
            f1s.append(f1)
            accs.append(acc)
        return np.mean(accs), np.mean(pres), np.mean(recalls), np.mean(f1s)


class ExplainModelNodeMultiFeature(torch.nn.Module):
    """
    explain BA-shapes and CiteSeer
    """
    def __init__(self, graph, base_model, target_node, args):
        super(ExplainModelNodeMultiFeature, self).__init__()
        self.graph = graph
        self.num_nodes = len(self.graph.nodes())
        self.feat = self.graph.ndata['feat']
        self.feat_dim = self.feat.shape[1]
        self.base_model = base_model
        self.target_node = target_node
        self.args = args
        self.feat_mask = self.construct_feat_mask()

    def forward(self):
        masked_feat = self.get_masked_feat()  # masked adj is always the exp sub graph
        pred1 = self.base_model(self.graph, masked_feat.float(),
                                self.graph.edata['weight'], self.target_node)
        pred2 = self.base_model(self.graph, (self.feat - masked_feat).float(),
                                self.graph.edata['weight'],
                                self.target_node)
        return pred1, pred2

    def loss(self, pred1, pred2, pred_label, gam, lam, alp):
        relu = torch.nn.ReLU()
        f_next = torch.max(torch.cat((pred1[:pred_label],
                                      pred1[pred_label+1:])))
        cf_next = torch.max(torch.cat((pred2[:pred_label],
                                       pred2[pred_label+1:])))
        bpr1 = relu(gam + f_next - pred1[pred_label])
        bpr2 = relu(gam + pred2[pred_label] - cf_next)
        masked_feat = self.get_masked_feat()
        L1 = torch.linalg.norm(masked_feat)
        loss = L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
        return bpr1, bpr2, L1, loss

    def construct_feat_mask(self):
        """
        construct mask for feature vector
        :return:
        """
        mask = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, self.feat_dim))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.num_nodes + self.feat_dim)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def get_masked_feat(self):
        feat_mask = torch.sigmoid(self.feat_mask)
        masked_feat = self.feat * feat_mask
        return masked_feat
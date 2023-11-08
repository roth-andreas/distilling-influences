import itertools
from glob import glob
from typing import List

import numpy as np
import torch

from dataset import get_dataset
from models import StudentNet, TeacherNet


def compare_influence(model_list: List, dataloader, idx, teacher, eval_influence):
    influence_scores = []
    preds = []
    node_idxs = torch.tensor([])
    label_entropies = []
    for model in model_list:
        model_infl_scores = []
        model_infl_preds = []
        model_preds = []
        for batch in dataloader:
            print("Loading batch.")
            batch.to('cuda')
            batch.x.requires_grad = True
            output = model(batch.x, batch.edge_index)
            feat = model.out_feat
            if idx is None:
                model_preds.append((torch.sigmoid(output) > 0.5).cpu().detach())
            else:
                model_preds.append(output.argmax(dim=-1).cpu().detach())
            if eval_influence:
                infl_scores, infl_preds, node_idxs, label_entropies = calc_influence(batch, feat, idx)
                model_infl_scores.append(infl_scores.cpu().detach())

        if eval_influence:
            influence_scores.append(torch.cat(model_infl_scores))
        preds.append(torch.cat(model_preds))

    teacher_infl_scores = []
    teacher_preds = []
    if teacher is not None:
        for batch in dataloader:
            print("Loading batch.")
            batch.to('cuda')
            batch.x.requires_grad = True
            output = teacher(batch.x, batch.edge_index)
            feat = teacher.out_feat
            if idx is None:
                teacher_preds.append((torch.sigmoid(output) > 0.5).cpu().detach())
            else:
                teacher_preds.append(output.argmax(dim=-1).cpu().detach())
            if eval_influence:
                infl_scores, infl_preds, _, _ = calc_influence(batch, feat, idx)
                teacher_infl_scores.append(infl_scores.cpu().detach())

        if eval_influence:
            teacher_infl_scores = torch.cat(teacher_infl_scores)
        teacher_preds = torch.cat(teacher_preds)

    return calc_errors(influence_scores, preds, node_idxs, idx, teacher_infl_scores,
                       teacher_preds, label_entropies)


def calc_influence(data, output, idx, create_graph=False):
    idxs = []
    infl_distributions = []
    infl_preds = []
    label_entropies = []
    calc_entropy = len(data.y.shape) == 1
    if calc_entropy:
        k = max(data.y) + 1
    if idx is None:
        nodes_to_eval = range(len(output))
    else:
        nodes_to_eval = torch.tensor(range(len(output)))[idx]
    for v in nodes_to_eval:
        score = torch.autograd.grad(output[v].abs().mean(), data.x, retain_graph=True, create_graph=create_graph)[
            0]  # ),data.y[v]], data.y[v]
        labels = []
        # Calc neighboring influence distribution
        N_v = data.edge_index[1, data.edge_index[0] == v]
        infl_distr = [score[v].abs().mean(dim=0, keepdim=True), score[N_v].abs().mean(dim=1)]
        infl_distr = torch.cat(infl_distr)
        infl_distributions.append(infl_distr / torch.norm(infl_distr))  # len(infl_distr) *

        infl_preds.append(torch.argmax(infl_distr.abs(), dim=0))

        if calc_entropy:
            labels.append(data.y[v])
            for j in N_v:
                #        #if True and split['train'][j]:
                labels.append(data.y[j])
            labels = torch.tensor(labels)
            label_entropies.append(entropy(labels, k).cpu().detach())
        idxs.append([v] * len(infl_distr))

    infl_distributions = torch.cat(infl_distributions)
    infl_preds = torch.stack(infl_preds)
    idxs = torch.tensor(np.concatenate(idxs))
    label_entropies = torch.tensor(label_entropies)
    return infl_distributions, infl_preds, idxs, label_entropies


def entropy(samples, k):
    eps = 1e-7
    n = len(samples) + eps
    result = 0
    for i in range(k):
        p = torch.sum(samples == i) / n
        result += torch.sum(-p * torch.log(p + eps))
    return result


def calc_errors(influence_scores, model_preds, id_idxs, test_idxs, t_is, t_p, label_entropies):
    id, churn, id_s, id_u, en = pairwise_errors(influence_scores, model_preds,
                                                                         id_idxs, test_idxs, label_entropies)
    t_id = []
    t_churn = []
    t_id_s = []
    t_id_u = []
    t_en = []
    if len(t_p) > 0:
        for i in range(len(model_preds)):
            if len(influence_scores) == len(model_preds):
                id_out, churn_out, id_s_out, id_u_out, en_out = pairwise_errors(
                    [influence_scores[i], t_is], [model_preds[i], t_p], id_idxs, test_idxs,
                    label_entropies)
                t_id.append(id_out[0])
                t_churn.append(churn_out[0])
                if len(id_s_out) > 0:
                    t_id_s.append(id_s_out[0])
                    t_id_u.append(id_u_out[0])
                    t_en.append(en_out[0])
            else:
                _, churn_out, _, _, _ = pairwise_errors([], [model_preds[i], t_p], id_idxs, test_idxs,
                                                                    label_entropies)
                t_churn.append(churn_out[0])

    return id, churn, id_s, id_u, en, t_id, t_churn, t_id_s, t_id_u, t_en


def correlation(label_entropies, stability):
    return np.corrcoef(label_entropies, stability)


def pairwise_errors(influence_scores, model_preds, id_idxs, test_idxs, label_entropies):
    id, id_s, id_u, churn, en = [], [], [], [], []

    node_list = torch.arange((len(model_preds[0])))
    for i, j in itertools.combinations(np.arange(len(model_preds)), 2):
        if len(model_preds[i].shape) == 1:
            stable_nodes = model_preds[i] == model_preds[j]
            test_stable_nodes = stable_nodes[test_idxs]
            unstable_nodes = ~test_stable_nodes
            stable_nodes_ids = torch.isin(id_idxs, node_list[stable_nodes])
            unstable_nodes_ids = ~stable_nodes_ids
            if len(influence_scores) == len(model_preds):
                #id_s.append(l1loss(influence_scores[i][stable_nodes_ids], influence_scores[j][stable_nodes_ids]))
                #id_u.append(l1loss(influence_scores[i][unstable_nodes_ids], influence_scores[j][unstable_nodes_ids]))
                en.append(correlation(label_entropies, test_stable_nodes)[0,1])
                id_s.append(correlation(np.abs(influence_scores[i] - influence_scores[j]), stable_nodes_ids)[0,1])
                #torch.abs(a - b) / ((torch.abs(a) + torch.abs(b)) * 0.5)
                id_u.append(0)

        if len(influence_scores) == len(model_preds):
            id.append(l1loss(influence_scores[i], influence_scores[j]))
        churn.append(instability(model_preds[i], model_preds[j]))

    return id, churn, id_s, id_u, en


def instability(a, b):
    return 1 - torch.eq(a, b).sum() / len(a.view(-1))


def l1loss(a, b):
    return torch.mean(torch.abs(a - b) / ((torch.abs(a) + torch.abs(b)) * 0.5)).cpu().detach()


def calc_instabilities(model_paths, teacher_path, eval_influence):
    test_dataset = None
    test_loader = None
    accs = []

    model_list = []
    teacher = None
    for run in range(len(model_paths)):
        try:
            checkpoint = torch.load(f"{model_paths[run]}/results.pt")
            if test_dataset is None:
                _, _, test_dataset, _, _, test_loader, split_idx = get_dataset(
                    name=vars(checkpoint['args']).get('dataset', 'ppi'),
                    root='data',
                )
            model = StudentNet(test_dataset.num_features, test_dataset.num_classes,
                               h_channels=checkpoint['args'].hidden_channels,
                               conv=vars(checkpoint['args']).get('conv', 'gat')).to('cuda')
            if checkpoint['args'].training != 'supervised' or checkpoint['args'].do_drop:
                model.teacher = TeacherNet(test_dataset.num_features, test_dataset.num_classes,
                                           h_channels=vars(checkpoint['args']).get('teacher_h', 256)).to('cuda')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model_list.append(model)
            accs.append(checkpoint['Test'])

            print(f'Loaded {model_paths[run]}')
        except:
            continue
    if teacher_path is not None:
        checkpoint = torch.load(teacher_path)
        teacher = TeacherNet(test_dataset.num_features, test_dataset.num_classes,
                             h_channels=vars(checkpoint['args']).get('h_channels', 256)).to('cuda')
        teacher.load_state_dict(checkpoint['model_state_dict'])
        teacher.eval()

    if test_loader is None:
        test_loader = [test_dataset._data]
        split_idx = split_idx['test']
    infl_diff, churn, id_s, id_u, en, t_id, t_churn, t_id_s, t_id_u, t_en = compare_influence(
        model_list, test_loader, split_idx, teacher, eval_influence)
    print(infl_diff, churn)
    print(
        f'Teacher Acc:{checkpoint["Test"] if teacher_path is not None else 0.0:.3f}, Student Acc: {np.mean(accs)*100:.1f}+-{np.std(accs)*100:.1f}, Churn: {np.mean(churn)*100:.1f}+-{np.std(churn)*100:.1f}')
    if eval_influence:
        print(
            f'ID: {np.mean(infl_diff)*100:.1f}+-{np.std(infl_diff)*100:.1f}')
        print(
            f'Corr(ID,S): {np.mean(id_s):.3f}+-{np.std(id_s):.3f}')
        print(
            f'T_ID:{np.mean(t_id)*100:.1f}+-{np.std(t_id)*100:.1f}')
        print(
            f'Corr(T_ID,T_S): {np.mean(t_id_s):.3f}+-{np.std(t_id_s):.3f}')
        print(
            f'EN:{np.mean(en):.3f}+-{np.std(en):.3f}, TEN:{np.mean(t_en):.3f}+-{np.std(t_en):.3f}')
    if teacher_path is not None:
        print(f'T_Churn: {np.mean(t_churn)*100:.1f}+-{np.std(t_churn)*100:.1f}')

def run(eval_model, param_name, eval_teacher=False, eval_influence=False):
    base_path = f'logs/{eval_model}/{param_name}/'
    model_list = glob(f'{base_path}/*')
    #model_list = glob(base_path + '/*')

    teacher_path = None
    if eval_teacher:
        dataset = eval_model.partition('_')[0]
        teacher_path = f'./checkpoints/{dataset}_seed5/checkpoint.pt'

    calc_instabilities(model_list, teacher_path, eval_influence)
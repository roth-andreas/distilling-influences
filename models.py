import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from criterion import *
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes


def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly drops edges from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    or index indicating which edges were retained, depending on the argument
    :obj:`force_undirected`.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


class TeacherNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_channels=256, dropout=0.0):
        super(TeacherNet, self).__init__()
        heads = 4
        self.conv1 = GATConv(in_channels, h_channels, heads=heads)
        self.lin1 = torch.nn.Linear(in_channels, heads * h_channels)
        self.conv2 = GATConv(heads * h_channels, h_channels, heads=heads)
        self.lin2 = torch.nn.Linear(heads * h_channels, heads * h_channels)
        out_heads = 6
        self.conv3 = GATConv(heads * h_channels, out_channels, heads=out_heads, concat=False)
        self.lin3 = torch.nn.Linear(heads * h_channels, out_channels)

        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def layer1(self, x, edge_index):
        x = self.dropout(x)
        return F.elu(self.conv1(x, edge_index) + self.lin1(x))

    def layer2(self, x, edge_index):
        x = self.dropout(x)
        return F.elu(self.conv2(x, edge_index) + self.lin2(x))

    def layer3(self, x, edge_index):
        x = self.dropout(x)
        return self.conv3(x, edge_index) + self.lin3(x)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        self.out_feat = x
        x = self.layer3(x, edge_index)
        return x


class StudentNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_channels=64, init_teacher=False, drop_edge_p=0.2,conv='gat'):
        super(StudentNet, self).__init__()
        self.drop_edge_p = drop_edge_p

        if conv == 'gat':
            self.conv1 = GATConv(in_channels, h_channels, heads=2)
            self.conv2 = GATConv(2 * h_channels, h_channels, heads=2)
            self.conv3 = GATConv(2 * h_channels, out_channels, heads=2, concat=False)
            self.lin1 = torch.nn.Linear(in_channels, 2 * h_channels)
            self.lin2 = torch.nn.Linear(2 * h_channels, 2 * h_channels)
            self.lin3 = torch.nn.Linear(2 * h_channels, out_channels)
        elif conv == 'gcn':
            self.conv1 = GCNConv(in_channels, h_channels)
            self.conv2 = GCNConv(h_channels, h_channels)
            self.conv3 = GCNConv(h_channels, out_channels)
            self.lin1 = torch.nn.Linear(in_channels, h_channels)
            self.lin2 = torch.nn.Linear(h_channels, h_channels)
            self.lin3 = torch.nn.Linear(h_channels, out_channels)

        if init_teacher:
            self.teacher = TeacherNet(in_channels, out_channels)
        else:
            self.teacher = None

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x, edge_index, eval_teacher=False, p=0.0):
        edge_index_s, _ = dropout_edge(edge_index, p=p, force_undirected=False)
        student_x = x

        if self.training:
            edge_index, _ = dropout_edge(edge_index_s, p=self.drop_edge_p, force_undirected=False)
        student_x = F.elu(self.conv1(student_x, edge_index) + self.lin1(student_x))
        if self.training:
            edge_index, _ = dropout_edge(edge_index_s, p=self.drop_edge_p, force_undirected=False)
        student_x = F.elu(self.conv2(student_x, edge_index) + self.lin2(student_x))
        self.out_feat = student_x
        if self.training:
            edge_index, _ = dropout_edge(edge_index_s, p=self.drop_edge_p, force_undirected=False)
        student_x = self.conv3(student_x, edge_index) + self.lin3(student_x)

        if eval_teacher:
            with torch.no_grad():
                teacher_x = x
                edge_index_t = edge_index_s
                teacher_x = self.teacher.layer1(teacher_x, edge_index_t)
                teacher_x = self.teacher.layer2(teacher_x, edge_index_t)
                self.out_teacher = teacher_x
                teacher_x = self.teacher.layer3(teacher_x, edge_index_t)
            return student_x, teacher_x
        else:
            return student_x

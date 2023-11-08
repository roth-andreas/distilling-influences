import argparse
import time
import os
import random
from torch.utils.tensorboard import SummaryWriter

import instability
from logger import Logger
from criterion import *

import train_helpers
from dataset import get_dataset
from models import TeacherNet, StudentNet


def train(model, teacher_model, train_dataset, train_loader, optimizer, args, device, student_proj=None, teacher_proj=None, idx=None):
    model.train()
    if student_proj:
        student_proj.train()
    if teacher_proj:
        teacher_proj.train()
    if teacher_model:
        teacher_model.eval()

    avg_loss = 0
    avg_loss_cls = 0
    avg_loss_aux = 0

    if train_loader is not None:
        iterator = enumerate(train_loader)
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        iterator = enumerate([train_dataset._data])
        criterion = torch.nn.CrossEntropyLoss()
    for step, batch in iterator:
        batch = batch.to(device)
        labels = batch.y

        out = model(batch.x, batch.edge_index, eval_teacher=False)

        if train_loader is None:
            out = out[idx]
            labels = labels[idx]
            if len(labels.shape) > 1:
                labels = labels[:,0]

        if args.eta > 0:
            x = batch.x
            out_xi, teacher_out_xi = model(x, batch.edge_index, p=args.dd_p, eval_teacher=True)
            if train_loader is None:
                out_xi = out_xi[idx]
                teacher_out_xi = teacher_out_xi[idx]
            loss_grads = F.mse_loss(out_xi, teacher_out_xi)
            loss_cls = loss_grads
            loss_aux = loss_grads
            loss = loss_grads#(1-args.eta)* loss + args.eta*
        else:

            if args.training == 'supervised':
                loss = criterion(out, labels)
                loss_cls = loss
                loss_aux = loss*0
            else:
                with torch.no_grad():
                    teacher_out = teacher_model(batch.x, batch.edge_index)

                if args.training == 'kd':
                    with torch.no_grad():
                        if train_loader is None:
                            teacher_out = F.softmax(teacher_out[idx] / args.kd_T, dim=1)
                        else:
                            teacher_out = torch.sigmoid(teacher_out / args.kd_T)
                    loss, loss_cls, loss_aux = kd_criterion(
                        out, labels, teacher_out, criterion, args.alpha, args.kd_T
                    )
                else:
                    teacher_out_feat = teacher_model.out_feat
                    out_feat = model.out_feat
                    if train_loader is None:
                        teacher_out_feat = teacher_out_feat[idx]
                        out_feat = out_feat[idx]

                    if args.training == 'nce':
                        out_feat = student_proj(out_feat)
                        teacher_out_feat = teacher_proj(teacher_out_feat)
                        loss, loss_cls, loss_aux = nce_criterion(
                            out, labels, out_feat, teacher_out_feat, criterion,
                            args.beta, args.nce_T, args.max_samples
                        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item()
        avg_loss_cls += loss_cls.detach().item()
        avg_loss_aux += loss_aux.detach().item()

    avg_loss /= (step + 1)
    avg_loss_cls /= (step + 1)
    avg_loss_aux /= (step + 1) 
    return avg_loss, avg_loss_cls, avg_loss_aux


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, param_name):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, split_idx = get_dataset(
        name=args.dataset,
        root='data',
    )

    if args.gnn == 'teacher':
        model = TeacherNet(train_dataset.num_features, train_dataset.num_classes).to(device)
        args.heads = 4  # hardcode heads for student_proj
    elif args.gnn == 'student':
        model = StudentNet(train_dataset.num_features, train_dataset.num_classes, args.hidden_channels,
                           drop_edge_p=args.drop_edge_p, conv=args.conv).to(device)
        args.heads = 2  # hardcode heads for student_proj
    else:
        raise ValueError('Invalid GNN type')
    if args.conv != 'gat':
        args.heads = 1

    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')

    logger = Logger(args.runs, args)
    val_accs = []

    for run in range(args.runs):
        seed(args.seed + run)

        model.reset_parameters()
        
        teacher_model = None
        if (args.training != 'supervised') or args.do_drop:
            checkpoint = torch.load(f"{args.teacher_path}/{args.dataset}_seed{args.seed}/checkpoint.pt")
            args.teacher_h = vars(checkpoint['args']).get('h_channels', 256)

            teacher_model = TeacherNet(train_dataset.num_features, train_dataset.num_classes, args.teacher_h).to(device)
            teacher_model.load_state_dict(checkpoint['model_state_dict'])
            teacher_model.eval()
            model.teacher = TeacherNet(train_dataset.num_features, train_dataset.num_classes, args.teacher_h).to(device)
            model.teacher.load_state_dict(checkpoint['model_state_dict'])
            model.teacher.eval()

        if args.training in ["nce"]:
            student_proj = torch.nn.Sequential(
                torch.nn.Linear(args.hidden_channels* args.heads, args.proj_dim), 
                torch.nn.BatchNorm1d(args.proj_dim), 
                torch.nn.ReLU()
            ).to(device)

            teacher_proj = torch.nn.Sequential(
                torch.nn.Linear(args.teacher_h * 4, args.proj_dim),
                torch.nn.BatchNorm1d(args.proj_dim), 
                torch.nn.ReLU()
            ).to(device)

            optimizer = torch.optim.Adam([
                {'params': model.parameters(), 'lr': args.lr},
                {'params': student_proj.parameters(), 'lr': args.lr},
                {'params': teacher_proj.parameters(), 'lr': args.lr}
            ])
        else:
            student_proj, teacher_proj = None, None
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Create Tensorboard logger
        log_dir = os.path.join(
            "logs",
            args.expt_name,
            param_name,
            f"GPU{args.device}-seed{args.seed+run}"
        )
        tb_logger = SummaryWriter(log_dir)
        
        # Start training
        best_epoch = 0
        best_train = 0
        best_val = 0
        best_test = 0
        not_improved = 0
        if args.do_drop:
            args.eta = 1.0
            if args.only_dd:
                args.warmup_steps = 100000
        else:
            args.eta = 0.0

        for epoch in range(1, 1 + args.epochs):
            if args.do_drop and epoch == args.warmup_steps:
                if args.training in ["nce"]:
                    optimizer = torch.optim.Adam([
                        {'params': model.parameters(), 'lr': args.lr},
                        {'params': student_proj.parameters(), 'lr': args.lr},
                        {'params': teacher_proj.parameters(), 'lr': args.lr}
                    ])
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                args.eta = 0
            train_loss, train_loss_cls, train_loss_aux = train(
                model, teacher_model, train_dataset, train_loader, 
                optimizer, args, device, student_proj, teacher_proj, idx=split_idx['train'] if split_idx is not None else None
            )
            
            train_acc,_ = train_helpers.test(model, train_dataset, train_loader, device, idx=split_idx['train'] if split_idx is not None else None)
            valid_acc,valid_loss = train_helpers.test(model, val_dataset, val_loader, device, idx=split_idx['valid'] if split_idx is not None else None)
            test_acc,test_loss = train_helpers.test(model, test_dataset, test_loader, device, idx=split_idx['test'] if split_idx is not None else None)
            logger.add_result(run, (train_acc, valid_acc, test_acc))

            if epoch % args.log_steps == 0:
                #print(f'Run: {run + 1:02d}, '
                #      f'Epoch: {epoch:02d}, '
                #      f'Loss_total: {train_loss:.4f}, '
                #      f'Loss_cls: {train_loss_cls:.4f}, '
                #      f'Loss_aux: {train_loss_aux:.8f}, '
                #      f'Train: {100 * train_acc:.2f}%, '
                #      f'Valid: {100 * valid_acc:.2f}% '
                #      f'Test: {100 * test_acc:.2f}%')
                
                # Log statistics to Tensorboard, etc.
                tb_logger.add_scalar('loss/train', train_loss, epoch)
                tb_logger.add_scalar('loss/cls', train_loss_cls, epoch)
                tb_logger.add_scalar('loss/aux', train_loss_aux, epoch)
                tb_logger.add_scalar('acc/train', train_acc, epoch)
                tb_logger.add_scalar('acc/valid', valid_acc, epoch)
                tb_logger.add_scalar('acc/test', test_acc, epoch)
                tb_logger.add_scalar('loss/valid', valid_loss, epoch)
                tb_logger.add_scalar('loss/test', test_loss, epoch)

            if valid_acc > best_val:
                best_epoch = epoch
                best_train = train_acc
                best_val = valid_acc
                best_test = test_acc
                not_improved = 0
                #print('--------- New Best ----------------')
            else:
                not_improved += 1
                if not_improved > 400 and epoch > min(args.warmup_steps + 200,1700):
                    break

        logger.print_statistics(run)
        torch.save({
            'args': args,
            'total_param': total_param,
            'BestEpoch': best_epoch,
            'Train': best_train,
            'Validation': best_val, 
            'Test': best_test,
            'model_state_dict': model.state_dict(),
        }, os.path.join(log_dir, "results.pt"))
        val_accs.append(best_val)

    return np.mean(val_accs)

def grid_search(args):
    best_val = 0.0
    best_param_name = ""
    if args.dataset == 'ppi':
        args.hidden_channels = 32
        steps = [300]
        alphas = [0.5]
        betas = [0.1]
        if args.drop_edge:
            drop_rates = [0.2]
        else:
            drop_rates = [0]
    else:
        if args.drop_edge:
            drop_rates = [0.2,0.4]
        else:
            drop_rates = [0]
        if args.do_drop:
            steps = [50,800,1500]#
        else:
            steps = [0]
        if args.training == 'kd':
            alphas = [0.25,0.5]
        else:
            alphas = [0]
        if args.training == 'nce':
            betas = [0.03,0.1,0.3]
        else:
            betas = [0]

    for beta in betas:
        for alpha in alphas:
            for drop_edge_p in drop_rates:#0.2
                for warmup_steps in steps:#, 600
                    for dd_p in [0.2]:
                            args.beta = beta
                            args.alpha = alpha
                            start_time_str = time.strftime("%Y%m%dT%H%M%S")
                            param_name = f"{args.conv}-{args.gnn}-{beta}-{alpha}-{drop_edge_p}-{warmup_steps}-{dd_p}-{start_time_str}"
                            args.drop_edge_p = drop_edge_p
                            args.warmup_steps = warmup_steps
                            args.dd_p = dd_p
                            val = main(args, param_name)
                            if val > best_val:
                                best_val = val
                                best_param_name = param_name
    print(f'Best Model was {best_param_name} with val acc {best_val:.3f}')
    return best_param_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPI')
    
    # Experiment settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=5, help="seed")
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--training', type=str, default="supervised")
    parser.add_argument('--expt_name', type=str, default="debug")
    parser.add_argument('--teacher_path', type=str, default="checkpoints")

    # GNN settings
    parser.add_argument('--gnn', type=str, default='student')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.005)#0.005
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--runs', type=int, default=5)
    
    # KD settings
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha parameter for KD (default: 0.5)')
    parser.add_argument('--kd_T', type=float, default=1.0,
                        help='temperature parameter for KD (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='beta parameter for auxiliary distillation losses (default: 0.0)')
    
    # NCE/auxiliary distillation settings
    parser.add_argument('--nce_T', type=float, default=0.075,
                        help='temperature parameter for NCE (default: 0.075)')
    parser.add_argument('--max_samples', type=int, default=8192,
                        help='maximum samples for NCE/GPW (default: 8192)')
    parser.add_argument('--proj_dim', type=int, default=256,
                        help='common projection dimensionality for NCE/FitNet (default: 256)')
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='kernel for LPW: cosine, polynomial, l2, rbf (default: rbf)')

    parser.add_argument('--do_drop', action='store_true')
    parser.add_argument('--dataset', type=str, default='ppi')
    parser.add_argument('--warmup_steps', type=int, default=300)
    parser.add_argument('--only_dd', action='store_true')
    parser.add_argument('--drop_edge', action='store_true')
    parser.add_argument('--conv', type=str, default='gat')

    args = parser.parse_args()

    print(args)
    best_param_name = grid_search(args)

    instability.run(args.expt_name, best_param_name, True)


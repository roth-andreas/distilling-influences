import argparse
import os
import random
from torch.utils.tensorboard import SummaryWriter

from logger import Logger

from criterion import *
import train_helpers
from dataset import get_dataset
from models import TeacherNet


def train(model, train_dataset, train_loader, optimizer, args, device, idx):
    model.train()
    
    avg_loss = 0

    if train_loader is not None:
        iterator = enumerate(train_loader)
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        iterator = enumerate([train_dataset._data])
        criterion = torch.nn.CrossEntropyLoss()
    
    for step, batch in iterator:
        batch = batch.to(device)
        labels = batch.y

        out = model(batch.x, batch.edge_index)

        if train_loader is None:
            out = out[idx]
            labels = labels[idx]
            if len(labels.shape) > 1:
                labels = labels[:,0]

        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item()
        
    avg_loss /= (step + 1)
    return avg_loss


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def main():
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, split_idx = get_dataset(
        name=args.dataset,
        root='data',
    )

    model = TeacherNet(train_dataset.num_features, train_dataset.num_classes, args.h_channels, dropout=0.0 if args.dataset == 'ppi' else 0.5).to(device)
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        seed(args.seed + run)

        model.reset_parameters()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Create Tensorboard logger
        log_dir = os.path.join(
            "checkpoints",
            f"{args.dataset}_seed{args.seed+run}"
        )
        tb_logger = SummaryWriter(log_dir)
        
        # Start training
        best_epoch = 0
        best_train = 0
        best_val = np.inf
        best_test = 0
        not_improved = 0
        best_train_loss = np.inf
        for epoch in range(1, 1 + args.epochs):
            train_loss = train(
                model, train_dataset, train_loader, 
                optimizer, args, device, split_idx['train'] if split_idx is not None else None
            )
            
            train_acc,_ = train_helpers.test(model, train_dataset, train_loader, device, idx=split_idx['train'] if split_idx is not None else None)
            valid_acc,valid_loss = train_helpers.test(model, val_dataset, val_loader, device, idx=split_idx['valid'] if split_idx is not None else None)
            test_acc,_ = train_helpers.test(model, test_dataset, test_loader, device, idx=split_idx['test'] if split_idx is not None else None)
            logger.add_result(run, (train_acc, valid_acc, test_acc))

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {train_loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                
                # Log statistics to Tensorboard, etc.
                tb_logger.add_scalar('loss/train', train_loss, epoch)
                tb_logger.add_scalar('acc/train', train_acc, epoch)
                tb_logger.add_scalar('acc/valid', valid_acc, epoch)
                tb_logger.add_scalar('acc/test', test_acc, epoch)

                if valid_loss < best_val:
                    best_epoch = epoch
                    best_train = train_acc
                    best_val = valid_loss
                    best_test = test_acc

                    torch.save({
                        'args': args,
                        'total_param': total_param,
                        'BestEpoch': best_epoch,
                        'Train': best_train,
                        'Validation': best_val, 
                        'Test': best_test,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(log_dir, "checkpoint.pt"))
                if train_loss < best_train_loss:
                    not_improved = 0
                    best_train_loss = train_loss
                else:
                    not_improved += 1
                    if not_improved > 10:
                        for g in optimizer.param_groups:
                            g['lr'] = g['lr'] / 3
                        #break
                        not_improved = 0
                        #print("Reduced LR!")

        logger.print_statistics(run)
        
    logger.print_statistics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPI')
    
    # Experiment settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=5, help="seed")
    parser.add_argument('--log_steps', type=int, default=1)
    
    # GNN settings
    parser.add_argument('--lr', type=float, default=0.005)#0.01
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--dataset', type=str, default='ppi')
    parser.add_argument('--h_channels', type=int, default=256)

    args = parser.parse_args()

    if args.dataset in ['citeseer','photo','ppi']:
        args.h_channels = 256
    elif args.dataset in ['computers', 'wikics']:
        args.h_channels = 128
    else:
        args.h_channels = 64

    print(args)
    main()
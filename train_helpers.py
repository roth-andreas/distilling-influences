import numpy as np
import torch
from sklearn.metrics import f1_score

@torch.no_grad()
def test(model, dataset, loader, device, idx):
    model.eval()

    if loader is not None:
        criterion = torch.nn.BCEWithLogitsLoss()
        ys, preds = [], []
        losses = []
        for data in loader:
            ys.append(data.y)
            out = model(data.x.to(device), data.edge_index.to(device))
            preds.append((out > 0).float().cpu())
            losses += [criterion(out, data.y.cuda()).cpu().detach().item()]

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0, np.mean(losses)
    else:
        out = model(dataset._data.x, dataset._data.edge_index)
        y_pred = out.argmax(dim=-1, keepdim=True)
        acc = (y_pred.view(-1)[idx] == dataset._data.y.view(-1)[idx]).sum().item() / dataset._data.y[idx].numel()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out[idx], dataset._data.y[idx])
        return acc, loss
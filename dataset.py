import os
from typing import List, Callable, Dict

import torch
import torch_geometric
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import WikiCS, Planetoid, Coauthor, Amazon, PPI
from torch_geometric.loader import DataLoader


def get_dataset(
        name: str,
        root: str,
        num_train_per_class: int = 20,
):
    if name.lower() == 'ppi':
        train_dataset = PPI('data/PPI/', split='train')
        val_dataset = PPI('data/PPI/', split='val')
        test_dataset = PPI('data/PPI/', split='test')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        split_idx = None
    else:
        # Now differentiate between different datasets
        if name.lower() == "wikics":
            dataset = WikiCS(os.path.join(root, "wikics"))
        elif name.lower() == "citeseer":
            dataset = Planetoid(root=root, name="CiteSeer", split="public")
        else:

            torch.manual_seed(0)
            split = T.RandomNodeSplit(
                split="test_rest",
                num_splits=1,
                num_train_per_class=num_train_per_class,
                num_val=500,
            )
            # dataset is from Coauthor or Amazon -> no predefined train-val-test split
            transform = T.Compose([split])

            if name.lower() == "physics":
                dataset = Coauthor(root, name="Physics", transform=transform)
            elif name.lower() == "photo":
                dataset = Amazon(root, name="Photo", transform=transform)
            elif name.lower() == "computers":
                dataset = Amazon(root, name="Computers", transform=transform)
            else:
                raise ValueError(f"Unknown dataset: {name}")
        split_idx = get_idx_split(dataset)
        test_dataset = val_dataset = train_dataset = dataset
        train_loader = val_loader = test_loader = None

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, split_idx


def get_idx_split(dataset: torch_geometric.data.Dataset) -> Dict[str, torch.Tensor]:
    data = dataset[0]
    if isinstance(dataset, PygNodePropPredDataset):
        idx: Dict[str, torch.Tensor] = dataset.get_idx_split()  # type:ignore
        # convert each index of variable length with node ids into a boolean vector with fixed length
        for key, tensor in idx.items():
            new_tensor = torch.zeros((data.num_nodes,), dtype=torch.bool)  # type:ignore
            new_tensor[tensor] = True
            idx[key] = new_tensor
    else:
        idx = {
            "train": data.train_mask,  # type: ignore
            "valid": data.val_mask,  # type: ignore
            "test": data.test_mask,  # type: ignore
        }
    # If there are multiple datasplits (mainly WikiCS), then .{train,val,test}_mask has
    # shape [num_nodes, num_splits]. For ease of use, remove all but the first one.
    for mask_name, mask in idx.items():
        assert isinstance(mask, torch.Tensor)
        if mask.ndim > 1:
            idx[mask_name] = mask[:, 0]
    return idx  # type:ignore

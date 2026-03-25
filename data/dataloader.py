import torch
from torch_geometric.loader import DataLoader


def get_dataloaders(train_set, test_set, batch_size=2, debug=False):

    if debug:
        print("DEBUG MODE: batch_size = 1")

        return (
            DataLoader(train_set, batch_size=1, shuffle=False),
            DataLoader(test_set, batch_size=1, shuffle=False),
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,   # IMPORTANT (single test graph)
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader
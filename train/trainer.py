# trainer.py
import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, graphs, lr=0.001, epochs=200, device=None, verbose=True):
        self.model = model
        self.graphs = graphs
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.graphs = [g.to(self.device) for g in self.graphs]

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0

            for graph in self.graphs:
                self.optimizer.zero_grad()
                pred = self.model(graph)

                # Use train_mask if exists
                mask = getattr(graph, "train_mask", None)
                if mask is not None:
                    pred_masked = pred[mask]
                    target_masked = graph.y[mask]
                else:
                    pred_masked = pred
                    target_masked = graph.y

                loss = F.mse_loss(pred_masked, target_masked)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.graphs)
            if self.verbose and (epoch == 1 or epoch % 10 == 0):
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

        if self.verbose:
            print("\nTraining finished!")

        return self.model
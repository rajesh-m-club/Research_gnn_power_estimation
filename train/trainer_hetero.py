import torch
import torch.nn.functional as F


class HeteroTrainer:

    def __init__(self, model, train_loader, test_loader=None, lr=1e-3):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------
    # Train one epoch
    # ------------------------------------------------
    def train_epoch(self):

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:

            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(batch)   # [N_net, 1]

            mask = batch["net"].train_mask
            target = batch["net"].y

            target = target.view_as(pred)

            if mask.sum() == 0:
                continue

            pred_masked = pred[mask]
            target_masked = target[mask]

            loss = F.mse_loss(pred_masked, target_masked)

            loss.backward()

            # Gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_samples += mask.sum().item()

        if total_samples == 0:
            return 0.0

        return total_loss / total_samples

    # ------------------------------------------------
    # Evaluate (USES eval_mask)
    # ------------------------------------------------
    def evaluate(self):

        if self.test_loader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():

            for batch in self.test_loader:

                batch = batch.to(self.device)

                pred = self.model(batch)

                # IMPORTANT FIX
                mask = batch["net"].eval_mask
                target = batch["net"].y

                target = target.view_as(pred)

                if mask.sum() == 0:
                    continue

                pred_masked = pred[mask]
                target_masked = target[mask]

                loss = F.mse_loss(pred_masked, target_masked)

                total_loss += loss.item() * mask.sum().item()
                total_samples += mask.sum().item()

                all_preds.append(pred_masked.view(-1))
                all_targets.append(target_masked.view(-1))

        if total_samples == 0:
            return 0.0, 0.0, 0.0

        # -------------------------
        # Aggregate metrics
        # -------------------------
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        mse = total_loss / total_samples
        mae = torch.mean(torch.abs(all_preds - all_targets)).item()

        # Correlation (safe)
        if all_preds.numel() > 1:
            corr = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1].item()
        else:
            corr = 0.0

        return mse, mae, corr

    # ------------------------------------------------
    # Debug batch
    # ------------------------------------------------
    def debug_batch(self, batch):

        print("\n--- DEBUG BATCH ---")

        print("Net nodes:", batch["net"].num_nodes)
        print("Trainable:", batch["net"].train_mask.sum().item())

        if hasattr(batch["net"], "eval_mask"):
            print("Eval nodes:", batch["net"].eval_mask.sum().item())

        print("Pred sample:", self.model(batch)[:5].view(-1))
        print("Target sample:", batch["net"].y[:5].view(-1))

    # ------------------------------------------------
    # Full training loop
    # ------------------------------------------------
    def train(self, epochs=20):

        for epoch in range(epochs):

            train_loss = self.train_epoch()

            eval_results = self.evaluate()

            if eval_results is not None:
                mse, mae, corr = eval_results

                print(
                    f"Epoch {epoch+1}: "
                    f"Train Loss={train_loss:.6f}, "
                    f"Val MSE={mse:.6f}, "
                    f"MAE={mae:.6f}, "
                    f"Corr={corr:.4f}"
                )
            else:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}")
import torch


class ErrorCalculator:

    @staticmethod
    def compute_metrics(pred, target):

        pred = pred.cpu()
        target = target.cpu()

        # --------------------------
        # MAE
        # --------------------------

        mae = torch.mean(torch.abs(pred - target))

        # --------------------------
        # RMSE
        # --------------------------

        rmse = torch.sqrt(torch.mean((pred - target) ** 2))

        # --------------------------
        # Pearson Correlation
        # --------------------------

        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)

        numerator = torch.sum((pred - pred_mean) * (target - target_mean))

        denominator = torch.sqrt(
            torch.sum((pred - pred_mean) ** 2) *
            torch.sum((target - target_mean) ** 2)
        )

        if denominator == 0:
            corr = torch.tensor(0.0)
        else:
            corr = numerator / denominator

        return {
            "MAE": mae.item(),
            "RMSE": rmse.item(),
            "Correlation": corr.item()
        }
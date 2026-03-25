import matplotlib.pyplot as plt


class PredictionPlotter:

    @staticmethod
    def plot(pred, target, save_path="predicted_vs_true.png"):

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        plt.figure(figsize=(6, 6))

        plt.scatter(target, pred, alpha=0.5)

        plt.xlabel("True Toggle (Gate SAIF)")
        plt.ylabel("Predicted Toggle (GNN)")
        plt.title("Predicted vs True Toggle Rates")

        # ideal line
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())

        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.grid(True)

        plt.savefig(save_path)

        print(f"Plot saved to {save_path}")

        plt.close()
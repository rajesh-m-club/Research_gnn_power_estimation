import os
import torch

from graph.build_graph_hetero import GraphBuilderHetero
from data.dataloader import get_dataloaders

# UPDATED MODEL IMPORT
from models.gnn_hetero_v3 import ToggleHeteroGNN_v3

from train.trainer_hetero import HeteroTrainer

from utils.error_calculation import ErrorCalculator
from utils.plot_prediction import PredictionPlotter


def main():

    # -----------------------------
    # Paths
    # -----------------------------
    netlist_dir = "data/netlists"
    gate_saif_dir = "data/saif/gate"
    rtl_saif_dir = "data/saif/rtl"
    cell_lib_file = "data/cell_library.json"

    # -----------------------------
    # Build dataset
    # -----------------------------
    builder = GraphBuilderHetero(
        netlist_dir,
        gate_saif_dir,
        cell_lib_file
    )

    test_netlist_name = sorted(os.listdir(netlist_dir))[-1]

    train_set, test_set = builder.build_dataset(test_netlist_name)

    if len(train_set) == 0:
        print("No training graphs found")
        return

    print("\nTraining graphs:", len(train_set))
    print("Testing graphs :", len(test_set))

    # -----------------------------
    # DataLoader
    # -----------------------------
    train_loader, test_loader = get_dataloaders(
        train_set,
        test_set,
        batch_size=2,
        debug=False
    )

    # -----------------------------
    # Feature dimensions
    # -----------------------------
    sample = train_set[0]

    net_dim = sample["net"].x.shape[1]
    pin_in_dim = sample["pin_in"].x.shape[1]
    pin_out_dim = sample["pin_out"].x.shape[1]
    cell_dim = sample["cell"].x.shape[1]

    print("\nFeature dimensions:")
    print("Net     :", net_dim)
    print("Pin_in  :", pin_in_dim)
    print("Pin_out :", pin_out_dim)
    print("Cell    :", cell_dim)

    # -----------------------------
    # Model (V3)
    # -----------------------------
    model = ToggleHeteroGNN_v3(
        net_feat_dim=net_dim,
        pin_feat_dim=pin_in_dim,
        cell_feat_dim=cell_dim,
        hidden_dim=64,
        num_layers=4,
        dropout=0.2
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = HeteroTrainer(
        model,
        train_loader,
        test_loader,
        lr=3e-3  # slightly lower for stability with max aggregation
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train(epochs=400)

    torch.save(model.state_dict(), "toggle_hetero_model_v3.pt")
    print("Model saved")

    # =========================================================
    # RTL → Gate INFERENCE
    # =========================================================
    print("\nRunning RTL → Gate inference...")

    netlist_path = os.path.join(netlist_dir, test_netlist_name)

    rtl_saif_file = test_netlist_name.replace("_netlist.v", "_rtl.saif")
    rtl_saif_path = os.path.join(rtl_saif_dir, rtl_saif_file)

    gate_saif_file = test_netlist_name.replace(".v", ".saif")
    gate_saif_path = os.path.join(gate_saif_dir, gate_saif_file)

    if not os.path.exists(rtl_saif_path):
        print(f"RTL SAIF missing: {rtl_saif_path}")
        return

    if not os.path.exists(gate_saif_path):
        print(f"Gate SAIF missing: {gate_saif_path}")
        return

    print("Using RTL SAIF :", rtl_saif_path)
    print("Using Gate SAIF:", gate_saif_path)

    # -----------------------------
    # Build graphs
    # -----------------------------
    rtl_data = builder.build_single_graph(
        netlist_path,
        rtl_saif_path,
        mode="test"
    )

    gate_data = builder.build_single_graph(
        netlist_path,
        gate_saif_path,
        mode="train"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    rtl_data = rtl_data.to(device)
    gate_data = gate_data.to(device)

    # -----------------------------
    # Inference
    # -----------------------------
    model.eval()

    with torch.no_grad():

        pred = model(rtl_data)

        # -----------------------------
        # Internal nets ONLY
        # -----------------------------
        train_mask = gate_data["net"].train_mask
        internal_mask = train_mask

        if internal_mask.sum() == 0:
            print("No internal nets for evaluation")
            return

        pred_masked = pred[internal_mask]
        target_masked = gate_data["net"].y[internal_mask]

        target_masked = target_masked.view_as(pred_masked)

        # -----------------------------
        # Metrics
        # -----------------------------
        metrics = ErrorCalculator.compute_metrics(
            pred_masked,
            target_masked
        )

        print("\nRTL → Gate comparison (INTERNAL NETS)")
        print("--------------------------------------")
        print("MAE  :", metrics["MAE"])
        print("RMSE :", metrics["RMSE"])
        print("Corr :", metrics["Correlation"])

        # -----------------------------
        # Plot
        # -----------------------------
        plot_file = "predicted_vs_true_hetero_v3.png"

        PredictionPlotter.plot(
            pred_masked,
            target_masked,
            plot_file
        )

        print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    main()
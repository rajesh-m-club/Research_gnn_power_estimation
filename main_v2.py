# main_v2.py
from graph.build_graph import GraphBuilder
from models.gnn_model2 import ToggleGNNv2
from train.trainer import Trainer

from utils.error_calculation import ErrorCalculator
from utils.plot_prediction import PredictionPlotter

import torch
import os


def main():
    # -----------------------------
    # Paths
    # -----------------------------
    netlist_dir = "data/netlists"
    gate_saif_dir = "data/saif/gate"
    rtl_saif_dir = "data/saif/rtl"
    cell_lib_file = "data/cell_library.json"

    # -----------------------------
    # Build training dataset
    # -----------------------------
    train_builder = GraphBuilder(netlist_dir, gate_saif_dir, cell_lib_file)
    graphs = train_builder.build_dataset()

    if len(graphs) < 2:
        print("Need at least 2 graphs to train/test.")
        return

    # -----------------------------
    # Train/Test split
    # -----------------------------
    train_graphs = graphs[:-1]
    test_gate_graph = graphs[-1]

    print("\nTraining circuits:", len(train_graphs))
    print("Test circuit nodes:", test_gate_graph.num_nodes)

    # -----------------------------
    # Build RTL graph for testing
    # -----------------------------
    test_netlist_file = sorted(os.listdir(netlist_dir))[-1]
    netlist_path = os.path.join(netlist_dir, test_netlist_file)
    rtl_saif_file = test_netlist_file.replace("_netlist.v", "_rtl.saif")
    rtl_saif_path = os.path.join(rtl_saif_dir, rtl_saif_file)

    print("Using RTL SAIF:", rtl_saif_path)
    test_graph = train_builder.build_single_graph(netlist_path, rtl_saif_path)

    # -----------------------------
    # Initialize ToggleGNNv2 model
    # -----------------------------
    model = ToggleGNNv2(
        in_channels=train_graphs[0].x.shape[1],  # feature vector length from FeatureBuilder
        hidden_channels=64,
        num_logic_families=26,  # adjust according to your cell library
        embed_dim=8
    )

    # -----------------------------
    # Train model
    # -----------------------------
    trainer = Trainer(
        model=model,
        graphs=train_graphs,
        lr=0.001,
        epochs=400
    )
    trainer.train()

    # -----------------------------
    # Test model on gate-level graph
    # -----------------------------
    print("\nRunning gate-level inference on test circuit...")
    device = trainer.device
    model.eval()
    test_graph = test_graph.to(device)

    with torch.no_grad():
        pred = model(test_graph)

        # Use train_mask if exists, otherwise all nodes
        mask = getattr(test_graph, "train_mask", None)
        if mask is not None:
            pred_masked = pred[mask]
            target_masked = test_gate_graph.y[mask].to(device)
        else:
            pred_masked = pred
            target_masked = test_gate_graph.y.to(device)

        # -----------------------------
        # Compute error metrics
        # -----------------------------
        metrics = ErrorCalculator.compute_metrics(pred_masked, target_masked)
        print("\nRTL → Gate comparison")
        print("-----------------------------")
        print("MAE  :", metrics["MAE"])
        print("RMSE :", metrics["RMSE"])
        print("Corr :", metrics["Correlation"])

        # -----------------------------
        # Plot predictions
        # -----------------------------
        plot_file = "predicted_vs_true_v2.png"
        PredictionPlotter.plot(pred_masked, target_masked, plot_file)
        print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    main()
from graph.build_graph import GraphBuilder
from models.gnn_model import ToggleGNN
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

    train_builder = GraphBuilder(
        netlist_dir,
        gate_saif_dir,
        cell_lib_file
    )

    graphs = train_builder.build_dataset()

    if len(graphs) < 2:
        print("Need at least 2 graphs.")
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

    test_netlist = sorted(os.listdir(netlist_dir))[-1]

    netlist_path = os.path.join(netlist_dir, test_netlist)

    rtl_saif_name = test_netlist.replace("_netlist.v", "_rtl.saif")
    rtl_saif_path = os.path.join(rtl_saif_dir, rtl_saif_name)

    print("Using RTL SAIF:", rtl_saif_path)


    test_graph = train_builder.build_single_graph(
        netlist_path,
        rtl_saif_path
    )


    # -----------------------------
    # Initialize model
    # -----------------------------

    model = ToggleGNN(
        in_channels=train_graphs[0].x.shape[1],
        hidden_channels=64
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
    # Test model
    # -----------------------------

    print("\nRunning RTL inference on test circuit...")

    device = trainer.device
    model.eval()

    test_graph = test_graph.to(device)

    with torch.no_grad():

        pred = model(test_graph)

        mask = test_graph.train_mask

        pred = pred[mask]
        target = test_gate_graph.y[mask].to(device)

        # -----------------------------
        # Error metrics
        # -----------------------------

        metrics = ErrorCalculator.compute_metrics(pred, target)

        print("\nRTL → Gate comparison")
        print("-----------------------------")
        print("MAE  :", metrics["MAE"])
        print("RMSE :", metrics["RMSE"])
        print("Corr :", metrics["Correlation"])


        # -----------------------------
        # Plot predictions
        # -----------------------------

        PredictionPlotter.plot(pred, target)


if __name__ == "__main__":
    main()
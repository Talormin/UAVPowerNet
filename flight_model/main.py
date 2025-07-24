import numpy as np
import pandas as pd
from config.hyperparameters import ModelParams, TrainingParams, DataParams
from core.data_processing import FlightDataProcessor
from core.training import ModelTrainer
from core.evaluation import ModelEvaluator
from utils.visualization import ResultVisualizer


def main():
    # Initialize configurations
    data_config = DataParams()
    model_config = ModelParams()
    train_config = TrainingParams()

    # Data preparation
    data_processor = FlightDataProcessor(data_config, train_config)
    processed_data = data_processor.load_and_prepare_data("data/flight_data.csv")

    # Initialize components
    model_trainer = ModelTrainer(model_config, train_config)
    model_evaluator = ModelEvaluator()
    visualizer = ResultVisualizer()

    # Train and evaluate models for each flight mode
    all_results = {}
    combined_results = {
        "all_y_true": [],
        "all_y_pred": [],
        "loss": [],
        "val_loss": []
    }

    for mode, (x_train, y_train) in processed_data['train_data'].items():
        print(f"\nTraining model for flight mode: {mode}")

        # Get validation data
        x_val, y_val = processed_data['test_data'][mode]

        # Hyperparameter optimization
        best_params = model_trainer.optimize_hyperparameters(x_train, y_train, x_val, y_val)
        print(f"Best parameters for mode {mode}: {best_params}")

        # Train final model
        model, history = model_trainer.train_model(x_train, y_train, x_val, y_val, best_params)

        # Evaluate model
        metrics = model_evaluator.evaluate_model(
            model, x_val, y_val, x_train, y_train,
            processed_data['scalers'][1]
        )
        all_results[mode] = metrics

        # Store predictions for combined analysis
        y_pred = model.predict(x_val)
        y_pred_inv = processed_data['scalers'][1].inverse_transform(y_pred)
        y_val_inv = processed_data['scalers'][1].inverse_transform(y_val)

        combined_results["all_y_true"].extend(y_val_inv.flatten())
        combined_results["all_y_pred"].extend(y_pred_inv.flatten())
        combined_results["loss"].extend(history.history['loss'])
        combined_results["val_loss"].extend(history.history['val_loss'])

    # Convert to numpy arrays
    combined_results["all_y_true"] = np.array(combined_results["all_y_true"])
    combined_results["all_y_pred"] = np.array(combined_results["all_y_pred"])

    # Display and visualize results
    results_df = model_evaluator.aggregate_results(all_results)
    print("\nModel Evaluation Results:")
    print(results_df)

    print("\nAverage Metrics Across All Flight Modes:")
    print(results_df.mean(axis=0))

    # Visualize combined results
    visualizer.plot_combined_results(combined_results)


if __name__ == "__main__":
    main()
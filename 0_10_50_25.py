import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.metrics import r2_score

# Load the response matrix CSV file
file_path = 'RMBipolar02082023_081612.csv'
response_matrix = pd.read_csv(file_path)

# Normalize the response matrix (optional but recommended)
response_matrix.iloc[:, 1:] = (response_matrix.iloc[:, 1:] - response_matrix.iloc[:, 1:].min()) / (
        response_matrix.iloc[:, 1:].max() - response_matrix.iloc[:, 1:].min()
)

# Create scaled identity matrices
identity_matrix_1 = np.eye(response_matrix.shape[1] - 1) * 0.1
identity_matrix_2 = np.eye(response_matrix.shape[1] - 1) * 0.5
identity_matrix_3 = np.eye(response_matrix.shape[1] - 1) * 0.25


# Prediction function incorporating all identity matrices
def predict_beam_positions(steering_index):
    """
    Predict changes in beam positions based on the steering index and all identity matrices.
    """
    try:
        # Adjust for 1-based index
        steering_index_0_based = steering_index - 1

        # Extract the baseline row from the response matrix
        baseline = response_matrix.iloc[steering_index_0_based, 1:].values

        # Add contributions from all identity matrices
        adjustment_1 = identity_matrix_1[steering_index_0_based % identity_matrix_1.shape[0]]
        adjustment_2 = identity_matrix_2[steering_index_0_based % identity_matrix_2.shape[0]]
        adjustment_3 = identity_matrix_3[steering_index_0_based % identity_matrix_3.shape[0]]

        # Combine adjustments to form the predicted values
        adjusted_baseline = baseline + adjustment_1 + adjustment_2 + adjustment_3

        # Simulate predictions (or return adjusted values directly)
        simulated_predictions = adjusted_baseline
        return simulated_predictions
    except IndexError as e:
        print(f"Error: Invalid steering index {steering_index} - {e}")
        return np.zeros(response_matrix.shape[1] - 1)


# Function to calculate actual change from the response matrix
def calculate_actual_change(steering_index):
    """
    Get the actual changes in beam positions from the response matrix.
    """
    try:
        # Adjust for 1-based index
        steering_index_0_based = steering_index - 1
        return response_matrix.iloc[steering_index_0_based, 1:].values
    except IndexError as e:
        print(f"Error: Invalid steering index {steering_index} - {e}")
        return np.zeros(response_matrix.shape[1] - 1)


# Function to calculate accuracy
def calculate_accuracy(selected_indices):
    """
    Calculate model accuracy based on R² value for selected indices.
    """
    actual_values = []
    predicted_values = []

    for index in selected_indices:
        actual_values.extend(calculate_actual_change(index))
        predicted_values.extend(predict_beam_positions(index))

    # Ensure actual and predicted values are valid for R² calculation
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    if actual_values.shape != predicted_values.shape:
        print("Shape mismatch between actual and predicted values!")
        return 0.0

    # Calculate R² score
    try:
        r2 = r2_score(actual_values, predicted_values)
    except ValueError as e:
        print(f"Error calculating R² score: {e}")
        return 0.0

    # Convert R² to percentage
    accuracy_percentage = max(0, min(100, r2 * 100))  # Ensure it's between 0 and 100
    return accuracy_percentage


# GUI Application (Unchanged)
class BeamPositionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Beam Position Change Visualization")

        # Title label
        ttk.Label(root, text="Beam Position Change Visualization", font=("Arial", 16)).pack(pady=10)

        # Steering Magnet Selection Frame
        self.selection_frame = ttk.Frame(root)
        self.selection_frame.pack(pady=10)

        ttk.Label(self.selection_frame, text="Select Steering Magnet Indices (1-88):").grid(row=0, column=0,
                                                                                            columnspan=2, pady=5)

        # Checkboxes for magnet selection
        self.checkbox_frame = ttk.Frame(self.selection_frame)
        self.checkbox_frame.grid(row=1, column=0, columnspan=2)

        self.check_vars = []
        for i in range(88):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(self.checkbox_frame, text=str(i + 1), variable=var)
            cb.grid(row=i // 10, column=i % 10, sticky="w")
            self.check_vars.append(var)

        # Button Controls
        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(pady=10)

        ttk.Button(self.button_frame, text="Select All", command=self.select_all).grid(row=0, column=0, padx=5)
        ttk.Button(self.button_frame, text="Reset", command=self.reset_selection).grid(row=0, column=1, padx=5)
        ttk.Button(self.button_frame, text="Plot", command=self.plot_changes).grid(row=0, column=2, padx=5)

        # Accuracy Label
        self.accuracy_label = ttk.Label(root, text="Model Accuracy: N/A", font=("Arial", 12))
        self.accuracy_label.pack(pady=10)

        # Canvas for matplotlib plot
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack()

    def select_all(self):
        """Select all checkboxes."""
        for var in self.check_vars:
            var.set(True)

    def reset_selection(self):
        """Deselect all checkboxes and reset the plot."""
        for var in self.check_vars:
            var.set(False)
        self.ax.clear()
        self.ax.text(
            0.5, 0.5, "Selections cleared. Ready to plot.",
            horizontalalignment="center", verticalalignment="center",
            transform=self.ax.transAxes
        )
        self.accuracy_label.config(text="Model Accuracy: N/A")
        self.canvas.draw()

    def plot_changes(self):
        """Plot changes for selected indices and display accuracy."""
        # Clear the plot
        self.ax.clear()

        # Get selected indices
        selected_indices = [i + 1 for i, var in enumerate(self.check_vars) if var.get()]

        if selected_indices:
            predicted_values = []
            actual_values = []

            for index in selected_indices:
                # Predict and calculate actual changes
                predicted_change = predict_beam_positions(index)
                actual_change = calculate_actual_change(index)

                # Store for accuracy calculation
                predicted_values.extend(predicted_change)
                actual_values.extend(actual_change)

                # Plot the results
                self.ax.plot(predicted_change, label=f"Prediction (Index {index})", linestyle="--", marker="o")
                self.ax.plot(actual_change, label=f"Actual (Index {index})", linestyle="-", marker="x")

            # Update plot labels
            self.ax.set_title("Beam Position Changes for Selected Indices")
            self.ax.set_xlabel("Beam Position Indicator Index")
            self.ax.set_ylabel("Change in Position")
            self.ax.legend()

            # Calculate and display accuracy
            accuracy = calculate_accuracy(selected_indices)
            self.accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}%")
        else:
            # Show message if no index is selected
            self.ax.text(
                0.5, 0.5, "No indices selected! Please select at least one index.",
                horizontalalignment="center", verticalalignment="center",
                transform=self.ax.transAxes
            )
            self.accuracy_label.config(text="Model Accuracy: N/A")

        # Update the canvas
        self.canvas.draw()


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = BeamPositionApp(root)
    root.mainloop()

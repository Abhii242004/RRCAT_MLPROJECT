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

# Create a scaled identity matrix (with 0.1 as scalar)
identity_matrix = np.eye(response_matrix.shape[1] - 1) * 0.1

# Prediction function incorporating the identity matrix
def predict_beam_positions(steering_index):
    try:
        baseline = response_matrix.iloc[steering_index, 1:].values
        adjusted_baseline = baseline + identity_matrix[steering_index % identity_matrix.shape[0]]
        return adjusted_baseline
    except IndexError as e:
        print(f"Error: Invalid steering index {steering_index} - {e}")
        return np.zeros(response_matrix.shape[1] - 1)

# Function to calculate the actual change from the response matrix
def calculate_actual_change(steering_index):
    try:
        return response_matrix.iloc[steering_index, 1:].values
    except IndexError as e:
        print(f"Error: Invalid steering index {steering_index} - {e}")
        return np.zeros(response_matrix.shape[1] - 1)

# Function to calculate accuracy
def calculate_accuracy(selected_indices):
    actual_values = []
    predicted_values = []

    for index in selected_indices:
        actual_values.extend(calculate_actual_change(index))
        predicted_values.extend(predict_beam_positions(index))

    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    if actual_values.shape != predicted_values.shape:
        print("Shape mismatch between actual and predicted values!")
        return 0.0

    try:
        r2 = r2_score(actual_values, predicted_values)
    except ValueError as e:
        print(f"Error calculating R² score: {e}")
        return 0.0

    accuracy_percentage = max(0, min(100, r2 * 100))
    return accuracy_percentage

# GUI Application
class BeamPositionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Beam Position Change Visualization")

        ttk.Label(root, text="Beam Position Change Visualization", font=("Arial", 16)).pack(pady=10)

        self.selection_frame = ttk.Frame(root)
        self.selection_frame.pack(pady=10)

        ttk.Label(self.selection_frame, text="Select Steering Magnet Indices (0-87):").grid(row=0, column=0,
                                                                                            columnspan=2, pady=5)

        self.checkbox_frame = ttk.Frame(self.selection_frame)
        self.checkbox_frame.grid(row=1, column=0, columnspan=2)

        self.check_vars = []
        for i in range(88):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(self.checkbox_frame, text=str(i), variable=var)
            cb.grid(row=i // 10, column=i % 10, sticky="w")
            self.check_vars.append(var)

        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(pady=10)

        ttk.Button(self.button_frame, text="Select All", command=self.select_all).grid(row=0, column=0, padx=5)
        ttk.Button(self.button_frame, text="Reset", command=self.reset_selection).grid(row=0, column=1, padx=5)
        ttk.Button(self.button_frame, text="Plot", command=self.plot_in_new_window).grid(row=0, column=2, padx=5)

        self.accuracy_label = ttk.Label(root, text="Model Accuracy: N/A", font=("Arial", 12))
        self.accuracy_label.pack(pady=10)

    def select_all(self):
        for var in self.check_vars:
            var.set(True)

    def reset_selection(self):
        for var in self.check_vars:
            var.set(False)
        self.accuracy_label.config(text="Model Accuracy: N/A")

    def plot_in_new_window(self):
        selected_indices = [i for i, var in enumerate(self.check_vars) if var.get()]

        if not selected_indices:
            self.accuracy_label.config(text="No indices selected!")
            return

        new_window = tk.Toplevel(self.root)
        new_window.title("Beam Position Changes")

        figure, ax = plt.subplots(figsize=(8, 4))
        canvas = FigureCanvasTkAgg(figure, new_window)
        canvas.get_tk_widget().pack()

        predicted_values = []
        actual_values = []

        for index in selected_indices:
            predicted_change = predict_beam_positions(index)
            actual_change = calculate_actual_change(index)

            predicted_values.extend(predicted_change)
            actual_values.extend(actual_change)

            ax.plot(predicted_change, label=f"Prediction (Index {index})", linestyle="--", marker="o")
            ax.plot(actual_change, label=f"Actual (Index {index})", linestyle="-", marker="x")

        ax.set_title("Beam Position Changes for Selected Indices")
        ax.set_xlabel("Beam Position Indicator Index")
        ax.set_ylabel("Change in Position")
        ax.legend()

        accuracy = calculate_accuracy(selected_indices)
        self.accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}%")

        canvas.draw()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = BeamPositionApp(root)
    root.mainloop()

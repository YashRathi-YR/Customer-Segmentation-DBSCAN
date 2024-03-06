import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QVBoxLayout, QPushButton

class ClusteringModel:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = None

    def fit(self, data):
        try:
            self.labels = self.model.fit_predict(data)
            print("Labels:", self.labels)
            print("Number of clusters:", len(set(self.labels)) - (1 if -1 in self.labels else 0))
        except Exception as e:
            print(f"Error during fitting: {e}")
            self.labels = None

def read_csv_file():
    app = QApplication([])  # Create a PyQt application instance
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_path, _ = QFileDialog.getOpenFileName(None, "Select CSV file", "", "CSV Files (*.csv);;All Files (*)", options=options)
    app.quit()  # Close the PyQt application instance

    try:
        df = pd.read_csv(file_path)
        print("Upload successful")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def select_columns(data_frame):
    print("Available columns:")
    for i, column in enumerate(data_frame.columns, start=1):
        print(f"{i}. {column}")

    selected_columns = input("Enter column numbers for clustering (comma-separated): ").split(',')
    try:
        selected_columns = [data_frame.columns[int(col) - 1] for col in selected_columns]
        return selected_columns
    except (ValueError, IndexError):
        print("Invalid column selection. Please try again.")
        return select_columns(data_frame)

class FileDialogApp(QWidget):
    def __init__(self):
        super().__init__()

        self.clustering_model = None

        layout = QVBoxLayout()

        button = QPushButton("Select CSV File")
        button.clicked.connect(self.show_dialog)
        layout.addWidget(button)

        self.setLayout(layout)

    def show_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_path:
            self.run_clustering(file_path)

    def run_clustering(self, file_path):
        # Read CSV file using the read_csv_file function
        data_frame = pd.read_csv(file_path)

        # Allow the user to select columns for clustering
        clustering_columns = select_columns(data_frame)

        # Select the specified columns for clustering
        features = data_frame[clustering_columns].values

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Create an instance of the ClusteringModel class
        self.clustering_model = ClusteringModel(eps=0.5, min_samples=5)

        # Fit the model on the standardized data
        self.clustering_model.fit(features_scaled)

        # Create a DataFrame with the original data and cluster labels
        df_with_labels = pd.concat([data_frame, pd.Series(self.clustering_model.labels, name='ClusterLabel')], axis=1)

        # Print details of each cluster group
        for label in set(self.clustering_model.labels):
            if label == -1:
                continue  # Skip outliers
            cluster_group = df_with_labels[df_with_labels['ClusterLabel'] == label]
            print(f"\nCluster Group {label}:")
            print(cluster_group[clustering_columns])

        # Plot the clustered data
        plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c='red', marker='o', edgecolors='black', label='Outliers')
        clustered_mask = self.clustering_model.labels != -1
        plt.scatter(features_scaled[clustered_mask, 0], features_scaled[clustered_mask, 1], c=self.clustering_model.labels[clustered_mask], cmap='viridis', marker='o', edgecolors='black', label='Clustered Points')

        plt.title('DBSCAN Clustering')
        plt.xlabel(clustering_columns[0])
        plt.ylabel(clustering_columns[1])
        plt.legend()
        plt.show()

if __name__ == "__main__":
    app = QApplication([])
    file_dialog_app = FileDialogApp()
    file_dialog_app.show()
    app.exec_()

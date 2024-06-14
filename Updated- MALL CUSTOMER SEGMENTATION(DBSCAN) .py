import sys
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, 
    QTableWidget, QTableWidgetItem, QMessageBox, QDialog, QCheckBox, 
    QDialogButtonBox, QSplitter, QHeaderView, QLabel, QPushButton
)
from PyQt5.QtCore import Qt

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

class ColumnSelectorDialog(QDialog):
    def __init__(self, columns):
        super().__init__()
        self.setWindowTitle("Select Columns for Clustering")
        self.layout = QVBoxLayout(self)
        self.checkboxes = []

        for column in columns:
            checkbox = QCheckBox(column)
            self.layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def get_selected_columns(self):
        return [checkbox.text() for checkbox in self.checkboxes if checkbox.isChecked()]

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_title("DBSCAN Clustering")
        self.ax.set_xlabel("Feature 1")
        self.ax.set_ylabel("Feature 2")

class ClusteringApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data_frame = None
        self.df_with_labels = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Customer Segmentation using DBSCAN')
        self.setGeometry(100, 100, 1000, 800)

        self.canvas = MplCanvas(self)
        self.table_widget = QTableWidget(self)

        # Create the buttons
        self.open_file_button = QPushButton("Select CSV File", self)
        self.open_file_button.clicked.connect(self.show_dialog)

        self.save_file_button = QPushButton("Save CSV", self)
        self.save_file_button.clicked.connect(self.save_clustered_data)
        self.save_file_button.setEnabled(False)

        # Set up the layout
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table_widget)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.open_file_button)
        layout.addWidget(splitter)
        layout.addWidget(self.save_file_button)
        self.setCentralWidget(container)

        self.show()

    def show_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_path:
            self.run_clustering(file_path)

    def run_clustering(self, file_path):
        try:
            self.data_frame = pd.read_csv(file_path)
            print("Upload successful")
            print(self.data_frame.head())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file: {e}")
            return

        column_selector = ColumnSelectorDialog(self.data_frame.columns)
        if column_selector.exec_() == QDialog.Accepted:
            clustering_columns = column_selector.get_selected_columns()
            if not clustering_columns:
                QMessageBox.warning(self, "Warning", "No columns selected for clustering")
                return
        else:
            return

        features = self.data_frame[clustering_columns].values

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        self.clustering_model = ClusteringModel(eps=0.5, min_samples=5)
        self.clustering_model.fit(features_scaled)

        self.df_with_labels = pd.concat([self.data_frame, pd.Series(self.clustering_model.labels, name='ClusterLabel')], axis=1)

        self.update_table(self.df_with_labels)
        self.plot_clustering(features_scaled, clustering_columns)

        self.save_file_button.setEnabled(True)

    def update_table(self, df):
        self.table_widget.setRowCount(df.shape[0])
        self.table_widget.setColumnCount(df.shape[1])
        self.table_widget.setHorizontalHeaderLabels(df.columns)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[row, col]))
                self.table_widget.setItem(row, col, item)

    def plot_clustering(self, features_scaled, clustering_columns):
        self.canvas.ax.clear()
        clustered_mask = self.clustering_model.labels != -1
        outliers_mask = self.clustering_model.labels == -1

        self.canvas.ax.scatter(features_scaled[outliers_mask, 0], features_scaled[outliers_mask, 1], c='red', marker='o', edgecolors='black', label='Outliers')
        scatter = self.canvas.ax.scatter(features_scaled[clustered_mask, 0], features_scaled[clustered_mask, 1], c=self.clustering_model.labels[clustered_mask], cmap='viridis', marker='o', edgecolors='black', label='Clustered Points')

        self.canvas.ax.set_title('DBSCAN Clustering')
        self.canvas.ax.set_xlabel(clustering_columns[0])
        self.canvas.ax.set_ylabel(clustering_columns[1])
        self.canvas.ax.legend(*scatter.legend_elements(), title="Clusters")

        self.canvas.draw()

    def save_clustered_data(self):
        if self.df_with_labels is not None:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV file", "", "CSV Files (*.csv);;All Files (*)", options=options)

            if file_path:
                try:
                    self.df_with_labels.to_csv(file_path, index=False)
                    QMessageBox.information(self, "Success", "File saved successfully")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not save file: {e}")
        else:
            QMessageBox.warning(self, "Warning", "No data to save")

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    clustering_app = ClusteringApp()
    sys.exit(app.exec_())

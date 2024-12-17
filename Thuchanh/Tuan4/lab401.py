import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)


class LinkPrediction:
    def __init__(self, dataset="karate"):
        """
        Khởi tạo với lựa chọn dataset.

        Parameters:
            dataset (str): Tên dataset ('karate', 'davis', 'florentine')
        """
        if dataset == "karate":
            self.G = nx.karate_club_graph()
        elif dataset == "les":
            self.G = nx.les_miserables_graph()
        else:
            raise ValueError("Dataset không hợp lệ")

    def analyze_network(self):
        """Phân tích cơ bản về mạng."""
        print("\n=== Phân tích mạng ===")
        print(f"Số lượng node: {self.G.number_of_nodes()}")
        print(f"Số lượng cạnh: {self.G.number_of_edges()}")
        print(f"Hệ số clustering trung bình: {nx.average_clustering(self.G):.3f}")

        plt.figure(figsize=(15, 5))

        # Vẽ cấu trúc mạng ban đầu
        plt.subplot(131)
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=500,
            font_size=8,
        )
        plt.title("Cấu trúc mạng ban đầu")

        # Vẽ phân phối bậc
        plt.subplot(132)
        degrees = [d for n, d in self.G.degree()]
        plt.hist(degrees, bins=max(degrees) - min(degrees) + 1, edgecolor="black")
        plt.title("Phân phối bậc")
        plt.xlabel("Bậc")
        plt.ylabel("Số lượng node")

        # Vẽ phân phối hệ số clustering
        plt.subplot(133)
        clustering_coeffs = list(nx.clustering(self.G).values())
        plt.hist(clustering_coeffs, bins=10, edgecolor="black")
        plt.title("Phân phối hệ số clustering")
        plt.xlabel("Hệ số clustering")
        plt.ylabel("Số lượng node")

        plt.tight_layout()
        plt.show()

    def prepare_data(self):
        """
        Chuẩn bị dữ liệu bằng cách giả lập dự đoán tương lai:
        1. Ẩn 20% cạnh hiện có (tập test) - giả sử đây là các liên kết sẽ xuất hiện trong tương lai
        2. Sử dụng 80% cạnh còn lại (tập train) để học và dự đoán
        """
        # Copy đồ thị gốc để giữ lại thông tin đầy đủ
        self.original_G = self.G.copy()

        # 1. Chia cạnh thành train/test
        all_edges = list(self.G.edges())
        train_edges, test_edges = train_test_split(
            all_edges, test_size=0.2, random_state=42
        )

        # 2. Tạo đồ thị train bằng cách xóa các cạnh test
        self.train_G = self.G.copy()
        self.train_G.remove_edges_from(test_edges)

        # 3. Lấy các cặp node không có cạnh từ đồ thị train
        non_edges = list(nx.non_edges(self.train_G))
        np.random.shuffle(non_edges)
        test_non_edges = non_edges[: len(test_edges)]

        # Lưu lại để sử dụng
        self.train_edges = train_edges
        self.test_edges = test_edges  # Các cạnh thật sự sẽ xuất hiện
        self.test_non_edges = test_non_edges  # Các cạnh thật sự không xuất hiện

        print("\n=== Thông tin về tập dữ liệu ===")
        print(f"Tổng số cạnh ban đầu: {len(all_edges)}")
        print(f"Số cạnh train: {len(train_edges)}")
        print(f"Số cạnh test (cạnh sẽ xuất hiện): {len(test_edges)}")
        print(f"Số cặp node test không có cạnh: {len(test_non_edges)}")

        # Vẽ đồ thị train và test
        plt.figure(figsize=(12, 5))
        pos = nx.spring_layout(self.original_G, seed=42)

        plt.subplot(121)
        nx.draw(
            self.train_G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="blue",
            node_size=500,
            font_size=8,
        )
        plt.title("Đồ thị train\n(80% cạnh)")

        plt.subplot(122)
        nx.draw(
            self.train_G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="blue",
            node_size=500,
            font_size=8,
        )
        nx.draw_networkx_edges(
            self.train_G, pos, edgelist=test_edges, edge_color="red", style="dashed"
        )
        plt.title("Đồ thị test\nXanh: Train, Đỏ: Cạnh cần dự đoán")

        plt.tight_layout()
        plt.show()

    def common_neighbors_score(self, u, v):
        """Tính điểm dựa trên số lượng láng giềng chung."""
        return len(list(nx.common_neighbors(self.train_G, u, v)))

    def jaccard_coefficient_score(self, u, v):
        """Tính hệ số Jaccard."""
        neighbors_u = set(self.train_G.neighbors(u))
        neighbors_v = set(self.train_G.neighbors(v))
        if len(neighbors_u | neighbors_v) == 0:
            return 0
        return len(neighbors_u & neighbors_v) / len(neighbors_u | neighbors_v)

    def adamic_adar_score(self, u, v):
        """Tính chỉ số Adamic/Adar."""
        return sum(
            1 / np.log(self.train_G.degree(w))
            for w in nx.common_neighbors(self.train_G, u, v)
        )

    def extract_features(self, edge_list):
        """Trích xuất đặc trưng cho các cạnh."""
        features = []
        for u, v in edge_list:
            features.append(
                [
                    self.common_neighbors_score(u, v),
                    self.jaccard_coefficient_score(u, v),
                    self.adamic_adar_score(u, v),
                ]
            )
        return np.array(features)

    def visualize_prediction_validation(self, predictions, method_name):
        """
        Vẽ đồ thị để so sánh dự đoán với thực tế:
        - Xanh: Cạnh trong tập train
        - Đỏ: Cạnh dự đoán đúng (có trong test)
        - Vàng: Cạnh dự đoán sai (không có trong test)
        """
        plt.figure(figsize=(12, 5))
        pos = nx.spring_layout(self.original_G, seed=42)

        # Đồ thị 1: Dự đoán
        plt.subplot(121)
        # Vẽ cạnh train
        nx.draw(
            self.train_G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="green",
            width=1,
            node_size=500,
            font_size=8,
        )

        # Vẽ cạnh dự đoán đúng và sai
        test_pairs = self.test_edges + self.test_non_edges
        predicted_edges = [
            test_pairs[i] for i in range(len(predictions)) if predictions[i] == 1
        ]

        correct_edges = [e for e in predicted_edges if e in self.test_edges]
        wrong_edges = [e for e in predicted_edges if e not in self.test_edges]

        nx.draw_networkx_edges(
            self.train_G,
            pos,
            edgelist=correct_edges,
            edge_color="red",
            style="dashed",
            width=2,
        )
        nx.draw_networkx_edges(
            self.train_G,
            pos,
            edgelist=wrong_edges,
            edge_color="yellow",
            style="dashed",
            width=2,
        )
        plt.title(f"{method_name}\nXanh: Train, Đỏ: Dự đoán đúng, Vàng: Dự đoán sai")

        # Đồ thị 2: Thực tế
        plt.subplot(122)
        nx.draw(
            self.train_G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="green",
            width=1,
            node_size=500,
            font_size=8,
        )
        nx.draw_networkx_edges(
            self.train_G,
            pos,
            edgelist=self.test_edges,
            edge_color="red",
            style="dashed",
            width=2,
        )
        plt.title("Thực tế\nXanh: Train, Đỏ: Test")

        plt.tight_layout()
        plt.show()

    def evaluate_methods(self):
        """Đánh giá các phương pháp dự đoán liên kết."""
        print("\n=== Đánh giá các phương pháp ===")

        # Chuẩn bị dữ liệu test
        X_test = self.extract_features(self.test_edges + self.test_non_edges)
        y_test = np.hstack(
            [np.ones(len(self.test_edges)), np.zeros(len(self.test_non_edges))]
        )

        # Đánh giá từng phương pháp
        methods = ["Common Neighbors", "Jaccard Coefficient", "Adamic/Adar"]
        results = []

        for i in range(3):
            # Tính ngưỡng (có thể điều chỉnh)
            scores = X_test[:, i]
            threshold = np.percentile(scores, 70)  # Lấy ngưỡng 70%

            # Dự đoán
            predictions = (scores > threshold).astype(int)

            # Tính các metrics
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions)
            rec = recall_score(y_test, predictions)

            results.append(
                {
                    "Phương pháp": methods[i],
                    "Accuracy": f"{acc:.3f}",
                    "Precision": f"{prec:.3f}",
                    "Recall": f"{rec:.3f}",
                }
            )

            # Vẽ kết quả dự đoán
            self.visualize_prediction_validation(predictions, methods[i])

        # Vẽ bảng kết quả
        df_results = pd.DataFrame(results)
        plt.figure(figsize=(10, 3))
        plt.axis("off")
        table = plt.table(
            cellText=df_results.values,
            colLabels=df_results.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title("Bảng đánh giá các phương pháp")
        plt.show()

    def run_experiment(self):
        """Chạy toàn bộ thử nghiệm."""
        print("=== Bắt đầu thử nghiệm ===")
        self.analyze_network()
        self.prepare_data()
        self.evaluate_methods()


if __name__ == "__main__":
    print("Chọn dataset:")
    print("1. Zachary's Karate Club")
    print("2. Les Miserables Network")

    choice = input("Nhập lựa chọn (1-2): ")
    dataset_map = {"1": "karate", "2": "les"}

    if choice in dataset_map:
        lp = LinkPrediction(dataset=dataset_map[choice])
        lp.run_experiment()
    else:
        print("Lựa chọn không hợp lệ!")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import community, cuts
import pandas as pd


class CommunityDetectionPipeline:
    def __init__(self, graph):
        """
        Khởi tạo pipeline với một đồ thị.
        """
        self.graph = graph
        self.results = {}

    def apply_girvan_newman(self):
        """
        Áp dụng thuật toán Girvan-Newman để phát hiện cộng đồng.
        """
        girvan_newman_generator = community.girvan_newman(self.graph)
        best_partition = max(
            girvan_newman_generator,
            key=lambda partition: community.modularity(self.graph, partition),
        )
        self.results["Girvan-Newman"] = best_partition

    def apply_label_propagation(self):
        """
        Áp dụng thuật toán Label Propagation để phát hiện cộng đồng.
        """
        partition = list(community.label_propagation_communities(self.graph))
        self.results["Label Propagation"] = partition

    def apply_louvain(self):
        """
        Áp dụng thuật toán Louvain để phát hiện cộng đồng.
        """
        partition = list(community.louvain_communities(self.graph))
        self.results["Louvain"] = partition

    def calculate_metrics(self, communities):
        """
        Tính toán các chỉ số đánh giá cho một phân hoạch cộng đồng.
        """
        # Loại bỏ các cộng đồng rỗng hoặc không hợp lệ
        valid_communities = [c for c in communities if len(c) > 0]

        conductance_values = []
        for c in valid_communities:
            try:
                value = cuts.conductance(self.graph, c)
                conductance_values.append(value)
            except ZeroDivisionError:
                # Nếu gặp lỗi chia cho 0, gán giá trị NaN hoặc bỏ qua
                conductance_values.append(np.nan)

        normalized_cut_values = []
        for c in valid_communities:
            try:
                value = cuts.normalized_cut_size(self.graph, c)
                normalized_cut_values.append(value)
            except ZeroDivisionError:
                # Nếu gặp lỗi chia cho 0, gán giá trị NaN hoặc bỏ qua
                normalized_cut_values.append(np.nan)

        metrics = {
            "num_communities": len(valid_communities),
            "modularity": community.modularity(self.graph, valid_communities),
            "conductance": np.nanmean(
                conductance_values
            ),  # Tính giá trị trung bình bỏ qua NaN
            "normalized_cut": np.nanmean(
                normalized_cut_values
            ),  # Tính giá trị trung bình bỏ qua NaN
        }
        return metrics

    def create_community_node_colors(self, communities):
        """
        Tạo màu sắc cho các nút dựa trên phân hoạch cộng đồng.
        """
        node_colors = []
        for node in self.graph:
            for idx, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(plt.cm.jet(idx / len(communities)))
                    break
        return node_colors

    def visualize_communities(self, communities, title, ax):
        """
        Vẽ đồ thị với các cộng đồng được phân hoạch.
        """
        node_colors = self.create_community_node_colors(communities)
        pos = nx.spring_layout(self.graph, k=0.3, iterations=50, seed=2)
        modularity = community.modularity(self.graph, communities)

        ax.set_title(
            f"{title}\n{len(communities)} communities (Modularity: {modularity:.3f})"
        )
        nx.draw(
            self.graph,
            pos=pos,
            node_size=500,
            node_color=node_colors,
            with_labels=True,
            font_size=10,
            ax=ax,
        )

    def run_pipeline(self):
        """
        Chạy toàn bộ pipeline trên đồ thị.
        """
        self.apply_girvan_newman()
        self.apply_label_propagation()
        self.apply_louvain()

    def compare_algorithms(self):
        """
        So sánh các thuật toán và hiển thị kết quả.
        """
        metrics_data = []
        for name, partition in self.results.items():
            metrics = self.calculate_metrics(partition)
            metrics_data.append(
                {
                    "Algorithm": name,
                    "Num Communities": metrics["num_communities"],
                    "Modularity": metrics["modularity"],
                    "Conductance": metrics["conductance"],
                    "Normalized Cut": metrics["normalized_cut"],
                }
            )

        metrics_df = pd.DataFrame(metrics_data)

        # Vẽ các biểu đồ so sánh
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # So sánh số lượng cộng đồng
        axes[0].bar(
            metrics_df["Algorithm"], metrics_df["Num Communities"], color="#8E44AD"
        )
        axes[0].set_title("Num Communities Comparison")
        axes[0].set_xlabel("Algorithm")
        axes[0].set_ylabel("Num Communities")

        # So sánh các chỉ số khác
        bar_width = 0.6
        x = np.arange(len(metrics_df["Algorithm"]))
        axes[1].bar(
            x - bar_width / 2,
            metrics_df["Modularity"],
            width=bar_width,
            label="Modularity",
            color="#F2D140",
        )
        axes[1].bar(
            x - bar_width / 2,
            metrics_df["Conductance"],
            width=bar_width,
            label="Conductance",
            color="#FF6347",
            bottom=metrics_df["Modularity"],
        )
        axes[1].bar(
            x - bar_width / 2,
            metrics_df["Normalized Cut"],
            width=bar_width,
            label="Normalized Cut",
            color="#48C9B0",
            bottom=metrics_df["Modularity"] + metrics_df["Conductance"],
        )

        axes[1].set_title("Comparison of Metrics")
        axes[1].set_xlabel("Algorithm")
        axes[1].set_ylabel("Value")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metrics_df["Algorithm"])
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def visualize_all_communities(self):
        """
        Hiển thị đồ thị các cộng đồng từ các thuật toán.
        """
        fig, axes = plt.subplots(1, len(self.results), figsize=(20, 6))
        for ax, (name, partition) in zip(axes, self.results.items()):
            self.visualize_communities(partition, name, ax)
        plt.tight_layout()
        plt.show()


# Tạo đồ thị giả lập
G = nx.erdos_renyi_graph(n=30, p=0.1, seed=42)

# Khởi tạo và chạy pipeline
pipeline = CommunityDetectionPipeline(G)
pipeline.run_pipeline()

# Hiển thị các cộng đồng
pipeline.visualize_all_communities()

# So sánh các thuật toán
pipeline.compare_algorithms()

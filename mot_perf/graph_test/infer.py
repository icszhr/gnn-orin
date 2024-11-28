import torch
import torch.nn as nn
import dgl
import numpy as np
import time
from scipy.sparse import load_npz

# 数据路径
graph_file = "./graph/graph_structure.npz"
feature_file = "./graph/node_features.bin"

# 加载图结构
csr_matrix = load_npz(graph_file)
graph = dgl.from_scipy(csr_matrix)
graph.ndata[dgl.NID] = torch.arange(graph.num_nodes())

# 加载节点特征
num_nodes = graph.num_nodes()
feature_dim = 128
with open(feature_file, "rb") as f:
    node_features = np.frombuffer(f.read(), dtype=np.float32).reshape(num_nodes, feature_dim)

# 动态推理参数
num_inference_rounds = 50
sample_size = 1000
fan_out = [15, 10, 5]  # 三层采样，每层采样邻居数量

# 定义图神经网络（GNN）模型
class GNN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GNN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(hidden_size, out_feats, allow_zero_in_degree=True)

    def forward(self, blocks, features):
        h = features
        for l, (conv, block) in enumerate(zip([self.conv1, self.conv2], blocks)):
            h = conv(block, h)
            if l != len([self.conv1, self.conv2]) - 1:
                h = torch.relu(h)
        return h

# 初始化GNN模型
hidden_size = 64
output_size = 10  # 假设输出为10个类别
model = GNN(in_feats=feature_dim, hidden_size=hidden_size, out_feats=output_size)

# 推理逻辑
for round_num in range(num_inference_rounds):
    # 记录开始时间
    start_time = time.time()

    # 随机采样起始节点
    sample_nodes = torch.randint(0, num_nodes, (sample_size,), dtype=torch.long)

    # 记录采样时间
    sampling_time_start = time.time()

    # 构建节点采样的子图（三层采样）
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fan_out)
    dataloader = dgl.dataloading.DataLoader(
        graph,
        sample_nodes,
        sampler,
        batch_size=sample_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # 记录采样时间
    sampling_time_end = time.time()
    print(f"Sampling time (Round {round_num + 1}): {sampling_time_end - sampling_time_start:.4f} seconds")

    # 遍历采样的子图并进行推理
    for input_nodes, output_nodes, blocks in dataloader:
        # 获取采样的节点特征
        sampled_node_features = node_features[input_nodes.numpy()]

        # 记录特征收集时间
        gathering_time_start = time.time()

        # 将特征转换为torch tensor
        sampled_node_features = torch.tensor(sampled_node_features, dtype=torch.float32)

        # 记录特征收集时间
        gathering_time_end = time.time()
        print(f"Gathering time (Round {round_num + 1}): {gathering_time_end - gathering_time_start:.4f} seconds")

        # 推理
        inference_time_start = time.time()
        output = model(blocks, sampled_node_features)
        inference_time_end = time.time()
        print(f"Inference time (Round {round_num + 1}): {inference_time_end - inference_time_start:.4f} seconds")

    # 记录总时间
    end_time = time.time()
    print(f"Total time for Round {round_num + 1}: {end_time - start_time:.4f} seconds\n")

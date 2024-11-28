import torch
import dgl
import psutil
import time

# 内存日志记录器
class MemoryLogger:
    def __init__(self, log_file):
        self.initial_cpu_memory = psutil.virtual_memory().used
        self.initial_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.memory_log = []
        self.log_file = log_file

    def log_cpu_memory(self):
        cpu_memory = psutil.virtual_memory().used
        memory_change = cpu_memory - self.initial_cpu_memory
        self.memory_log.append(f"CPU Memory Change: {memory_change / (1024**2):.2f} MB")
        self.initial_cpu_memory = cpu_memory

    def log_gpu_memory(self):
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            memory_change = gpu_memory - self.initial_gpu_memory
            self.memory_log.append(f"GPU Memory Change: {memory_change / (1024**2):.2f} MB")
            self.initial_gpu_memory = gpu_memory

    def log_tensor_allocation(self, tensor_name, tensor):
        self.memory_log.append(f"Allocated tensor {tensor_name} with shape {tensor.shape} and dtype {tensor.dtype}")
        self.log_gpu_memory()  # 记录显存变化
        self.log_cpu_memory()  # 记录内存变化

    def write_log(self):
        with open(self.log_file, 'a') as f:
            for entry in self.memory_log:
                f.write(entry + '\n')

# 定义一个简单的 GCN 层
class GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_feats, out_feats)

    def forward(self, graph, feat, memory_logger):
        memory_logger.log_tensor_allocation("Node Features", feat)

        # 图的节点数据更新
        graph.ndata['h'] = feat
        graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h'))
        h = graph.ndata['h']
        memory_logger.log_tensor_allocation("Aggregated Node Features", h)

        h = self.linear(h)  # 线性变换
        memory_logger.log_tensor_allocation("Transformed Node Features", h)

        return h

# 测试推理并记录内存使用情况
def test_gcn_inference(log_file):
    num_nodes = 1000  # 节点数
    num_edges = 5000  # 边数
    g = dgl.graph((torch.randint(0, num_nodes, (num_edges,)), torch.randint(0, num_nodes, (num_edges,))))
    g = dgl.add_self_loop(g)

    node_feats = torch.randn((num_nodes, 64))  # 64 维特征
    model = torch.nn.ModuleList([GCNLayer(64, 128), GCNLayer(128, 64), GCNLayer(64, 32)])

    memory_logger = MemoryLogger(log_file)

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i, layer in enumerate(model):
            node_feats = layer(g, node_feats, memory_logger)

    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.4f} seconds")
    memory_logger.write_log()  # 将日志写入文件

# 设置日志文件路径
log_file = 'memory_log.txt'

# 执行推理并记录内存使用情况
test_gcn_inference(log_file)

import sys
import os
import time
import torch
import dgl
import networkx as nx
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ogb.nodeproppred import DglNodePropPredDataset
from src.algorithms.serializer.bfs_serializer import BFSSerializer
from src.algorithms.compression.bpe_engine import BPEEngine

# ==========================================
# Mock Loaders
# ==========================================

class MockLoader:
    def __init__(self):
        pass

    def get_node_token(self, graph, node_id, ntype=None):
        # Generic dummy token
        return [1]

    def get_edge_token(self, graph, edge_id, etype=None):
        return [0]

    def get_graph_node_type_ids(self, g):
        return torch.zeros(g.num_nodes(), dtype=torch.long)

    def get_graph_edge_type_ids(self, g):
        return torch.zeros(g.num_edges(), dtype=torch.long)
        
    def get_most_frequent_edge_type(self):
        return "default"
        
    def get_node_type(self, graph, node_id):
        return "node"
        
    def get_edge_type(self, graph, edge_id, edge_type=None):
        return "edge"

    def get_token_readable(self, token_id):
        return str(token_id)
        
    def get_graph_node_token_ids(self, g):
        return torch.zeros((g.num_nodes(), 1), dtype=torch.long)

    def get_graph_edge_token_ids(self, g):
        return torch.zeros((g.num_edges(), 1), dtype=torch.long)
        
    def get_edge_type_id_by_name(self, name):
        return 0

class MockArxivLoader(MockLoader):
    def get_node_token(self, graph, node_id, ntype=None):
        if 'label' in graph.ndata:
            label = graph.ndata['label'][node_id].item()
            if np.isnan(label): return [0]
            return [int(label)]
        return [0]
        
    def get_graph_node_token_ids(self, g):
        if 'label' in g.ndata:
             labels = g.ndata['label']
             if labels.dim() == 1: labels = labels.unsqueeze(1)
             labels = torch.nan_to_num(labels, nan=0.0).long()
             return labels
        return super().get_graph_node_token_ids(g)

# ==========================================
# Data Generation / Loading
# ==========================================

def load_ogbn_arxiv():
    print("Loading ogbn-arxiv dataset...")
    try:
        dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='dataset')
        graph, labels = dataset[0]
        graph.ndata['label'] = labels
        # Convert to bidirected for better BFS coverage
        graph = dgl.to_bidirected(graph)
        return [graph]
    except Exception as e:
        print(f"Error loading ogbn-arxiv: {e}")
        return []

def generate_fake_code2(num_graphs=2000):
    print(f"Generating {num_graphs} fake graphs simulating ogbg-code2 (AST-like trees)...")
    graphs = []
    # ogbg-code2: avg nodes ~125
    for _ in range(num_graphs):
        n_nodes = np.random.randint(50, 200)
        # Create a random tree
        g_nx = nx.random_tree(n_nodes)
        g_dgl = dgl.from_networkx(g_nx)
        # Add self loops or ensure it works with DGL
        g_dgl = dgl.add_self_loop(g_dgl)
        graphs.append(g_dgl)
    return graphs

# ==========================================
# Benchmarking Logic
# ==========================================

def benchmark_dataset(dataset_name, graphs, loader_cls):
    print(f"\n{'#'*20} Benchmarking {dataset_name} {'#'*20}")
    
    total_nodes = sum(g.num_nodes() for g in graphs)
    total_edges = sum(g.num_edges() for g in graphs)
    print(f"Total Graphs: {len(graphs)}")
    print(f"Total Nodes: {total_nodes}")
    print(f"Total Edges: {total_edges}")
    
    # 1. Serialization
    print("--- Serialization (BFS) ---")
    serializer = BFSSerializer()
    loader = loader_cls()
    serializer.initialize_with_dataset(loader)
    
    graph_data_list = [{'dgl_graph': g} for g in graphs]
    
    t0 = time.time()
    # Use batch_serialize for multiple graphs, serialize for single
    if len(graphs) == 1:
        result = serializer.serialize(graph_data_list[0])
        results = [result]
    else:
        results = serializer.batch_serialize(graph_data_list, parallel=False, desc="Serializing")
    t_serialization = time.time() - t0
    
    # Collect all tokens
    all_tokens = []
    for res in results:
        seq, _ = res.get_sequence(0)
        all_tokens.extend(seq)
    
    total_seq_len = len(all_tokens)
    print(f"Serialization Time: {t_serialization*1000:.2f} ms")
    print(f"Total Sequence Length: {total_seq_len}")
    
    # 2. BPE
    print("--- BPE Encoding ---")
    # Train on the sequence(s)
    # For fair comparison, we train on the full concatenated sequence (or list of sequences)
    # We pass list of sequences to train
    all_seqs = [res.get_sequence(0)[0] for res in results]
    
    bpe_engine = BPEEngine(train_backend='cpp', encode_backend='cpp')
    
    # Train
    # Limit merges to avoid over-compression on dummy data
    # For real data, 10k-50k merges is common
    num_merges = 2000
    
    t_train_start = time.time()
    bpe_engine.train(all_seqs, num_merges=num_merges, min_frequency=2)
    t_train = time.time() - t_train_start
    
    # Encode
    bpe_engine.build_encoder()
    t_encode_start = time.time()
    encoded_seqs = bpe_engine.batch_encode(all_seqs)
    t_encode = time.time() - t_encode_start
    
    encoded_len = sum(len(s) for s in encoded_seqs)
    
    print(f"BPE Encoding Time: {t_encode*1000:.2f} ms")
    print(f"Compressed Length: {encoded_len}")
    
    # 3. Inference
    print("--- Inference ---")
    # Simulate inference on a sequence of length = 1/10 of Total Nodes (normalized)
    # We measure speed on a chunk of size X, then extrapolate to "per 1M nodes"
    
    # Target length for benchmark run: 10k tokens (fits in GPU)
    bench_len = 10000
    
    d_model = 512
    nhead = 8
    num_layers = 6
    vocab_size = bpe_engine.vocab_size + 100
    
    model = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
        num_layers=num_layers
    )
    embedding = torch.nn.Embedding(vocab_size, d_model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    embedding = embedding.to(device)
    
    input_seq = torch.randint(0, vocab_size, (1, bench_len)).to(device)
    
    # Warmup
    with torch.no_grad():
        try:
            _ = model(embedding(input_seq[:, :100]))
            if device.type == 'cuda': torch.cuda.synchronize()
        except: pass
        
    if device.type == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model(embedding(input_seq))
    if device.type == 'cuda': torch.cuda.synchronize()
    t_infer_bench = time.time() - t0
    
    print(f"Inference Time (for {bench_len} tokens): {t_infer_bench*1000:.2f} ms")
    
    # ================= CALCULATION =================
    # Normalize to per 1M nodes
    scale = 1_000_000 / total_nodes
    
    norm_serialization = t_serialization * 1000 * scale
    norm_bpe = t_encode * 1000 * scale
    
    # For inference:
    # "Per 1M nodes" -> implies processing the compressed sequence corresponding to 1M nodes
    # If compression ratio is 10x, then 1M nodes -> 100k tokens.
    # We measured time for 10k tokens. So we need to scale by 10.
    # Estimated tokens for 1M nodes = 100,000
    est_tokens_1m = 100_000
    norm_inference = (t_infer_bench * 1000) * (est_tokens_1m / bench_len)
    
    return {
        'dataset': dataset_name,
        'serialization_ms': norm_serialization,
        'bpe_ms': norm_bpe,
        'inference_ms': norm_inference
    }

def main():
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    results = []
    
    # 1. ogbn-arxiv
    graphs_arxiv = load_ogbn_arxiv()
    if graphs_arxiv:
        res = benchmark_dataset("ogbn-arxiv", graphs_arxiv, MockArxivLoader)
        results.append(res)
        
    # 2. ogbg-code2 (simulated)
    graphs_code2 = generate_fake_code2(2000) # ~250k nodes
    res = benchmark_dataset("ogbg-code2", graphs_code2, MockLoader)
    results.append(res)

    # 3. ogbn-products (simulated proxy)
    # Products is denser (avg degree ~30)
    print("\nGenerating fake ogbn-products proxy (denser graph)...")
    # Generate smaller proxy due to time, scale up results
    g_prod = dgl.rand_graph(50000, 1500000) # 50k nodes, 1.5M edges (deg=30)
    g_prod = dgl.add_self_loop(g_prod) 
    res = benchmark_dataset("ogbn-products", [g_prod], MockLoader)
    results.append(res)
    
    print("\n\n" + "="*80)
    print(f"{'Dataset':<15} | {'Serialization (ms)':<20} | {'BPE (ms)':<10} | {'Inference (ms)':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<15} | {r['serialization_ms']:<20.2f} | {r['bpe_ms']:<10.2f} | {r['inference_ms']:<15.2f}")
    print("="*80)
    print("Note: All times are normalized 'per 1M nodes'.")
    print("Inference time assumes 10x compression (1M nodes -> 100k tokens).")

if __name__ == "__main__":
    main()

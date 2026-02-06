"""
Training script for causal graph discovery.
Builds causal graph from NHANES data using CausalNex (if available) or correlation-based fallback.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import joblib
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing.pipeline import load_raw_data, clean_data

# Try to import CausalNex
try:
    from causalnex.structure import StructureModel
    from causalnex.structure.notears import from_pandas
    CAUSALNEX_AVAILABLE = True
except ImportError:
    CAUSALNEX_AVAILABLE = False


def build_correlation_graph(df: pd.DataFrame, threshold: float = 0.3) -> nx.DiGraph:
    """
    Build a causal graph using correlation-based heuristics.
    
    Args:
        df: DataFrame with features
        threshold: Minimum absolute correlation to create an edge
        
    Returns:
        NetworkX directed graph
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        # Return empty graph if not enough numeric features
        return nx.DiGraph()
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr().abs()
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for col in numeric_df.columns:
        G.add_node(col)
    
    # Add edges based on correlation threshold
    for i, col1 in enumerate(numeric_df.columns):
        for col2 in numeric_df.columns[i+1:]:
            corr_value = corr_matrix.loc[col1, col2]
            if corr_value >= threshold:
                # Create bidirectional edge (simplified causal structure)
                # In practice, you'd use domain knowledge or constraint-based methods
                G.add_edge(col1, col2, weight=float(corr_value))
                G.add_edge(col2, col1, weight=float(corr_value))
    
    return G


def build_causalnex_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build causal graph using CausalNex NOTEARS algorithm.
    
    Args:
        df: DataFrame with features
        
    Returns:
        NetworkX directed graph
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return nx.DiGraph()
    
    try:
        # Use NOTEARS to learn structure
        sm = from_pandas(
            numeric_df,
            w_threshold=0.3,  # Threshold for edge weights
            tabu_edges=None,
            tabu_parent_nodes=None,
            tabu_child_nodes=None
        )
        
        # Convert StructureModel to NetworkX graph
        G = sm.to_networkx()
        return G
    except Exception as e:
        print(f"⚠️  CausalNex failed: {e}")
        print("   Falling back to correlation-based graph...")
        return None


def build_causal_graph_from_data(data_path: str, output_path: str):
    """
    Build and save causal graph from dataset.
    
    Args:
        data_path: Path to CSV dataset
        output_path: Path to save graph
    """
    print("=" * 60)
    print("Building Causal Graph")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return
    
    # Load and clean data
    print(f"Loading dataset from {data_path}...")
    df = load_raw_data(data_path)
    df = clean_data(df)
    
    print(f"Dataset: {len(df)} samples, {df.shape[1]} features")
    
    # Try CausalNex first, fallback to correlation
    graph = None
    method = "unknown"
    
    if CAUSALNEX_AVAILABLE:
        print("\nAttempting to build graph with CausalNex...")
        graph = build_causalnex_graph(df)
        if graph is not None and len(graph.nodes()) > 0:
            method = "causalnex"
            print(f"✅ Built graph with CausalNex: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        else:
            print("   CausalNex graph empty or failed, using fallback...")
    
    # Fallback to correlation-based graph
    if graph is None or len(graph.nodes()) == 0:
        print("\nBuilding correlation-based graph...")
        graph = build_correlation_graph(df, threshold=0.3)
        method = "correlation"
        print(f"✅ Built correlation graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    if graph is None or len(graph.nodes()) == 0:
        print("⚠️  Warning: Could not build graph. Creating minimal graph with all features as nodes...")
        # Create minimal graph with all numeric features as nodes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        graph = nx.DiGraph()
        for col in numeric_cols:
            graph.add_node(col)
        method = "minimal"
    
    # Save graph
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    joblib.dump(graph, output_path)
    
    print(f"\n✅ Causal graph saved to {output_path}")
    print(f"   Method: {method}")
    print(f"   Nodes: {len(graph.nodes())}")
    print(f"   Edges: {len(graph.edges())}")
    
    # Print some node names for reference
    if len(graph.nodes()) > 0:
        print(f"\nSample nodes: {list(graph.nodes())[:5]}")


def main():
    """Main function."""
    # Paths
    nhanes_path = "data/raw/NHANES Blood Panel Dataset.csv"
    output_path = "models/causal_graph.joblib"
    
    build_causal_graph_from_data(nhanes_path, output_path)
    
    print("\n" + "=" * 60)
    print("Causal graph training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


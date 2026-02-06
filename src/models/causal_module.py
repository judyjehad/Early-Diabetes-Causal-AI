import pandas as pd
import numpy as np
import joblib
import networkx as nx
from copy import deepcopy

# Try to import optional causal libraries
try:
    import dowhy
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

try:
    import causalnex
    CAUSALNEX_AVAILABLE = True
except ImportError:
    CAUSALNEX_AVAILABLE = False


def load_causal_graph(path: str):
    """
    Loads a causal graph from a saved file.
    
    Args:
        path: Path to the saved graph file (pickle/joblib format)
        
    Returns:
        NetworkX graph object if successful, None otherwise
    """
    # Return None if path is None or empty
    if not path or path == "":
        return None
    
    # Return None if file does not exist
    import os
    if not os.path.exists(path):
        return None
    
    # Load the graph using joblib
    try:
        graph = joblib.load(path)
        return graph
    except Exception:
        return None


def get_causal_effects(patient_features: dict, graph=None, top_k: int = 5) -> dict:
    """
    Identifies top causal drivers for a patient's features.
    
    Uses graph-based ranking if a causal graph is available,
    otherwise falls back to z-score based heuristic.
    
    Args:
        patient_features: Dictionary of feature names to values
        graph: NetworkX graph object (optional)
        top_k: Number of top drivers to return
        
    Returns:
        Dictionary with method, top_drivers, and notes
    """
    # Convert patient features to Series for easier manipulation
    patient_series = pd.Series(patient_features)
    
    # Method 1: Use graph if available
    if graph is not None and isinstance(graph, nx.Graph):
        # Calculate degree centrality for each node
        degree_centrality = nx.degree_centrality(graph)
        
        # Filter to only features present in patient_features
        feature_scores = []
        for feature in patient_features.keys():
            if feature in degree_centrality:
                score = degree_centrality[feature]
                feature_scores.append({"feature": feature, "score": float(score)})
        
        # Sort by score descending and take top_k
        feature_scores.sort(key=lambda x: x["score"], reverse=True)
        top_drivers = feature_scores[:top_k]
        
        return {
            "method": "graph",
            "top_drivers": top_drivers,
            "notes": "Ranked by graph degree centrality"
        }
    
    # Method 2: Fallback to z-score heuristic (numeric features only)
    numeric_features = {}
    for feature, value in patient_features.items():
        try:
            numeric_value = float(value)
            numeric_features[feature] = numeric_value
        except (ValueError, TypeError):
            # Skip non-numeric features
            continue
    
    if len(numeric_features) == 0:
        return {
            "method": "heuristic",
            "top_drivers": [],
            "notes": "No numeric features available for z-score calculation"
        }
    
    # Calculate z-scores (assuming mean=0, std=1 as a simple heuristic)
    # In practice, you'd use actual population statistics
    numeric_series = pd.Series(numeric_features)
    z_scores = np.abs(numeric_series)
    
    # Create driver list
    feature_scores = []
    for feature, z_score in z_scores.items():
        feature_scores.append({"feature": feature, "score": float(z_score)})
    
    # Sort by absolute z-score descending and take top_k
    feature_scores.sort(key=lambda x: x["score"], reverse=True)
    top_drivers = feature_scores[:top_k]
    
    return {
        "method": "heuristic",
        "top_drivers": top_drivers,
        "notes": "Ranked by absolute feature values (z-score approximation)"
    }


def simulate_intervention(patient_features: dict, feature: str, new_value: float) -> dict:
    """
    Simulates an intervention by changing a feature value.
    
    Args:
        patient_features: Original dictionary of patient features
        feature: Name of the feature to modify
        new_value: New value to assign to the feature
        
    Returns:
        Dictionary with feature, old_value, new_value, and updated_features
    """
    # Make a deep copy to avoid modifying the original
    updated_features = deepcopy(patient_features)
    
    # Get old value if it exists
    old_value = updated_features.get(feature, None)
    
    # Set the new value
    updated_features[feature] = new_value
    
    return {
        "feature": feature,
        "old_value": old_value,
        "new_value": new_value,
        "updated_features": updated_features
    }


import numpy as np
from typing import Tuple, List
import networkx as nx
import importlib.util
import sys
import traceback
import uuid
from itertools import combinations

def count_unit_triangles(vertices: np.ndarray, edges: List[Tuple[int, int]]) -> int:
    """Count unit triangles (3 vertices all at unit distance)"""
    edge_set = set(edges) | {(j, i) for i, j in edges}
    triangles = 0
    
    for i, j, k in combinations(range(len(vertices)), 3):
        if (i, j) in edge_set and (j, k) in edge_set and (i, k) in edge_set:
            triangles += 1
    
    return triangles



def evaluate(program_path: str) -> dict:
    """
    Evaluate unit distance graph using core metrics:
    
    1. n_vertices: Number of vertices
    2. n_edges: Number of edges  
    3. avg_degree: Average degree (2 * edges / vertices)
    4. n_triangles: Number of unit triangles
    
    Scoring philosophy:
    - Triangles: Highest weight (local chromatic constraints)
    - Average degree: High weight (connectivity indicator)
    - Edges: Medium weight (absolute connectivity)
    - Vertices: Low weight (prefer smaller dense graphs)
    """
    try:
        # Load module
        module_name = f"program_module_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(module_name, program_path)
        if spec is None or spec.loader is None:
            return {
                "runs_successfully": 0.0,
                "error": "Failed to load program",
                "combined_score": 0.0
            }
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'solve'):
            return {
                "runs_successfully": 0.0,
                "error": "No solve() function found",
                "combined_score": 0.0
            }
        
        result = module.solve()
        
        # Handle result: either vertices list or (vertices, edges) tuple
        if isinstance(result, tuple) and len(result) == 2:
            # Legacy format: (vertices, edges)
            vertices, edges = result
        elif isinstance(result, list):
            # New format: list of sympy complex numbers
            import sympy as sp
            # Convert sympy complex to numpy array
            vertices = np.array([[float(sp.re(p)), float(sp.im(p))] for p in result])
            # Compute edges from vertices
            edges = []
            n = len(vertices)
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(vertices[i] - vertices[j])
                    if abs(dist - 1.0) < 1e-4:
                        edges.append((i, j))
        elif isinstance(result, np.ndarray):
            # Numpy array of vertices
            vertices = result
            # Compute edges from vertices
            edges = []
            n = len(vertices)
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(vertices[i] - vertices[j])
                    if abs(dist - 1.0) < 1e-4:
                        edges.append((i, j))
        else:
            return {
                "runs_successfully": 0.0,
                "error": f"solve() must return vertices (list/array) or (vertices, edges), got {type(result)}",
                "combined_score": 0.0
            }
        
        n_vertices = len(vertices)
        n_edges = len(edges)
        
        # Basic validation
        if n_vertices < 4 or n_edges < 4:
            return {
                "runs_successfully": 0.0,
                "n_vertices": n_vertices,
                "n_edges": n_edges,
                "error": "Graph too small",
                "combined_score": 0.0
            }
        
        # Validate edge distances
        valid_edges = 0
        for i, j in edges:
            if i >= n_vertices or j >= n_vertices:
                continue
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if abs(dist - 1.0) < 1e-4:
                valid_edges += 1
        
        edge_validity = valid_edges / max(n_edges, 1)
        
        if edge_validity < 0.95:
            return {
                "runs_successfully": 0.0,
                "n_vertices": n_vertices,
                "n_edges": n_edges,
                "edge_validity": edge_validity,
                "error": "Too many invalid edges",
                "combined_score": 0.0
            }
        
        # === Calculate core metrics ===
        
        # Average degree
        avg_degree = 2 * n_edges / n_vertices
        
        # Number of triangles
        n_triangles = count_unit_triangles(vertices, edges)
        
        # === Scoring with positive weights ===
        
        # Normalize each metric to [0, 1] range for fair weighting
        
        # Vertices: Prefer graphs with reasonable size (10-30 vertices)
        # Use logarithmic scale to avoid penalizing larger graphs too much
        vertex_score = 1.0 / (1.0 + abs(np.log10(n_vertices / 15.0)))
        
        # Edges: More edges generally better
        # Normalize: good range is 20-80 edges
        edge_score = min(n_edges / 50.0, 1.0)
        
        # Average degree: Higher is better (indicates more connections)
        # Good range: 3.0 - 6.0
        degree_score = min(avg_degree / 5.0, 1.0)
        
        # Triangles: More is better (indicates high local density)
        # Normalize to [0, 1], aim for 20+ triangles
        triangle_score = min(n_triangles / 30.0, 1.0)
        
        # === Combined score with optimized weights ===
        # Weight distribution (total = 20):
        # - Triangles: 8.0 (most valuable - chromatic constraints)
        # - Average degree: 6.0 (very important - connectivity)
        # - Edges: 4.0 (important - absolute size)
        # - Vertices: 2.0 (least important - size management)
        
        combined_score = (
            vertex_score * 2.0 +      # Size management
            edge_score * 4.0 +        # Absolute connectivity
            degree_score * 6.0 +      # Average connectivity
            triangle_score * 8.0      # Chromatic constraints (highest!)
        )
        
        # Bonus for valid edges
        combined_score *= edge_validity
        
        return {
            "runs_successfully": 1.0,
            # === Core Metrics ===
            "n_vertices": n_vertices,
            "n_edges": n_edges,
            "avg_degree": avg_degree,
            "n_triangles": n_triangles,
            # === Normalized Scores ===
            "vertex_score": vertex_score,
            "edge_score": edge_score,
            "degree_score": degree_score,
            "triangle_score": triangle_score,
            # === Validation ===
            "edge_validity": edge_validity,
            # === Final Score ===
            "combined_score": combined_score
        }
        
    except Exception as e:
        return {
            "runs_successfully": 0.0,
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
            "combined_score": 0.0
        }
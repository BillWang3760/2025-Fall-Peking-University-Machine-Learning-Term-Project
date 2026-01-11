import numpy as np
from typing import List, Tuple
import sympy as sp

def shift_points(points: list, offset: sp.Expr) -> list:
    """Translate the point set"""
    return [sp.simplify(p + offset) for p in points]


def rotate_points(points: list, center: sp.Expr, angle: sp.Expr) -> list:
    """Rotate a set of points around the center"""
    rotation_factor = sp.cos(angle) + sp.I * sp.sin(angle)
    return [sp.simplify(center + (p - center) * rotation_factor) for p in points]

def merge_points(base_points, new_points, tolerance=1e-5):
    """Merge point sets and avoid duplicates"""
    result = base_points.copy()
    for point in new_points:
        point_val = complex(point)
        found = False
        for existing in result:
            if abs(point_val - complex(existing)) < tolerance:
                if sp.simplify(point - existing) == 0:
                    found = True
                    break
        if not found:
            result.append(sp.simplify(point))
    return result

def find_unit_circle_intersections(p1: np.ndarray, p2: np.ndarray) -> List[np.ndarray]:
    """
    Find intersection points of two unit circles centered at p1 and p2.
    This is the key technique from de Grey's paper.
    """
    dist = np.linalg.norm(p2 - p1)
    if dist > 2.0 or dist < 1e-10:
        return []
    
    # Midpoint and perpendicular direction
    mid = (p1 + p2) / 2
    if dist > 1e-10:
        perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / dist
    else:
        return []
    
    # Height of intersection points above/below midline
    h = np.sqrt(max(0, 1.0 - (dist / 2) ** 2))
    
    return [mid + h * perp, mid - h * perp]

def add_vertices_by_circle_intersection(vertices: np.ndarray, max_new: int = 5) -> np.ndarray:
    """
    Add new vertices at unit circle intersections (de Grey method).
    For each pair of existing vertices at distance <= 2, find points at unit distance from both.
    """
    new_vertices = []
    n = len(vertices)
    
    for i in range(n):
        for j in range(i + 1, n):
            candidates = find_unit_circle_intersections(vertices[i], vertices[j])
            for candidate in candidates:
                # Check if candidate is not too close to existing vertices
                is_new = True
                for v in vertices:
                    if np.linalg.norm(candidate - v) < 1e-6:
                        is_new = False
                        break
                for v in new_vertices:
                    if np.linalg.norm(candidate - v) < 1e-6:
                        is_new = False
                        break
                
                if is_new and len(new_vertices) < max_new:
                    new_vertices.append(candidate)
    
    if new_vertices:
        return np.vstack([vertices, np.array(new_vertices)])
    return vertices

def create_hexagonal_layer(radius: float, center: np.ndarray = np.array([0.0, 0.0])) -> np.ndarray:
    """
    Create a hexagonal arrangement of vertices at given radius.
    Hexagonal symmetry appears in many unit distance graphs.
    """
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices
    vertices = []
    for angle in angles:
        vertices.append(center + radius * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(vertices)

def create_initial_triangle_base() -> np.ndarray:
    """
    Create an initial base using equilateral triangles.
    This provides a good starting point for building dense unit distance graphs.
    """
    sqrt3 = np.sqrt(3)
    
    # Start with two equilateral triangles sharing an edge
    vertices = np.array([
        [0.0, 0.0],              # Base vertex 1
        [1.0, 0.0],              # Base vertex 2 (distance 1 from v1)
        [0.5, sqrt3/2],          # Top of first triangle
        [0.5, -sqrt3/2],         # Bottom of second triangle
    ])
    
    return vertices

# EVOLVE-BLOCK-START
def solve(seed: int = 42) -> np.ndarray:
    """
    Generate unit distance graph using iterative construction:
    1. Start with an initial triangle base
    2. Add vertices at unit circle intersections
    3. Use hexagonal layers at strategic locations
    4. Build interconnected structures with circle intersections
    5. Merge duplicate vertices
    """
    np.random.seed(seed)
    
    # Start with initial triangle base
    vertices = create_initial_triangle_base()
    
    # Add vertices using circle intersections (de Grey method)
    vertices = add_vertices_by_circle_intersection(vertices, max_new=10)
    
    # Add another round of circle intersections on the expanded set
    vertices = add_vertices_by_circle_intersection(vertices, max_new=10)
    
    # Add a hexagonal layer at a strategic location
    # Place center at one of the vertices to create connections
    # Choose a vertex that's likely to create many unit distances
    if len(vertices) > 2:
        hex_center = vertices[2]  # Using the top vertex of the first triangle
    else:
        hex_center = vertices[0]
    hex_layer = create_hexagonal_layer(radius=1.0, center=hex_center)
    
    # Convert all vertices to sympy expressions for merging
    all_points = []
    # Add initial vertices
    for point in vertices:
        all_points.append(sp.sympify(complex(point[0], point[1])))
    
    # Add hex layer vertices
    for point in hex_layer:
        all_points.append(sp.sympify(complex(point[0], point[1])))
    
    # Merge duplicate points
    merged_points = merge_points([], all_points)
    
    # Convert back to numpy array for return
    result = []
    for point in merged_points:
        c = complex(point)
        result.append([c.real, c.imag])
    
    return np.array(result)
# EVOLVE-BLOCK-END
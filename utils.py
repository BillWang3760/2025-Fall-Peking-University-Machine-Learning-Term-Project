import sympy as sp


def distance(z1, z2):
    """Calculate exact distance between two complex expressions"""
    delta = z2 - z1
    return sp.Abs(delta)


def point_to_coordinate(point):
    """Convert complex expression to coordinate tuple"""
    return float(sp.re(point)), float(sp.im(point))


def all_pairs(items):
    """Return all unordered pairs from a list"""
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append((items[i], items[j]))
    return pairs


def all_pair_sums(items):
    """Return sums of all unordered pairs"""
    return [a + b for (a, b) in all_pairs(items)]


def contains_point(point, point_list):
    """Check if a point is already in a list"""
    point_val = complex(point)
    for existing_point in point_list:
        existing_val = complex(existing_point)
        if abs(point_val - existing_val) < 1e-10:
            if sp.simplify(point - existing_point) == 0:
                return True
    return False


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

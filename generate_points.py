from utils import *


def triangle():
    points = [
        0 * sp.I,
        1,
        (1 + sp.I * sp.sqrt(3)) / 2
    ]
    return points


def create_H() -> list:
    """
    Create the basic hexagon graph H.
    H consists of a regular hexagon of side length 1 and its center.
    This graph has 7 vertices and 12 edges.
    """
    w = (1 + sp.I * sp.sqrt(3)) / 2
    points = [0 * sp.I] + [w ** i for i in range(6)]
    return [sp.simplify(p) for p in points]


def create_J() -> list:
    """
    Create graph J: 31 vertices formed by 13 copies of H
    Structure: central H + 6 H's at distance 1 + 6 H's at distance sqrt(3)
    """
    H = create_H()
    w = (1 + sp.I * sp.sqrt(3)) / 2
    points = H.copy()
    # 6 copies of H at unit distance from center
    offset = 1
    for i in range(6):
        offset = offset * w
        shifted_H = shift_points(H, offset)
        points = merge_points(points, shifted_H)
    # 6 copies of H at distance sqrt(3) from center
    offset = (3 + sp.I * sp.sqrt(3)) / 2
    for i in range(6):
        offset = offset * w
        shifted_H = shift_points(H, offset)
        points = merge_points(points, shifted_H)
    return [sp.simplify(p) for p in points]


def create_K() -> list:
    J = create_J()
    rotated_J = rotate_points(J, 0, 2 * sp.asin(sp.S(1) / 4))
    points = merge_points(J, rotated_J)
    return [sp.simplify(p) for p in points]


def create_L() -> list:
    K = create_K()
    rotated_K = rotate_points(K, -2, 2 * sp.asin(sp.S(1) / 8))
    points = merge_points(K, rotated_K)
    return [sp.simplify(p) for p in points]


def create_M() -> list:
    # Step 1: Generate the 30 unit vectors for graph V
    V = []
    base = sp.asin(sp.sqrt(3) / 2)
    offset = sp.asin(1 / sp.sqrt(12))
    for i in range(6):
        for j in range(-2, 3):
            angle = i * base + j * offset
            v = sp.exp(sp.I * angle)
            V.append(sp.simplify(v))
    V = merge_points([], V)
    print(f"Graph V completed: {len(V)} vertices")
    # Step 2: Create graph W = {sum of two vectors from V | distance from origin ≤ √3}
    W = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            p = sp.simplify(V[i] + V[j])
            if sp.simplify(sp.re(p * sp.conjugate(p)) - 3) <= 0:
                W.append(p)
    W = merge_points([], W)
    print(f"Graph W completed: {len(W)} vertices")
    # Step 3: Create M by translating W to the 6 vertices of H
    w = (1 + sp.I * sp.sqrt(3)) / 2
    hexagon_vertices = [w ** i for i in range(6)]
    M = W.copy()
    for vertex in hexagon_vertices:
        translated_W = shift_points(W, vertex)
        M = merge_points(M, translated_W)
    return W


def create_N():
    # TODO
    pass


def create_G() -> list:
    S = [
        sp.Rational(0, 1),  # (0, 0)
        sp.Rational(1, 3),  # (1/3, 0)
        sp.Rational(1, 1),  # (1, 0)
        sp.Rational(2, 1),  # (2, 0)
        (sp.sqrt(33) - 3) / 6,    # ((√33 - 3)/6, 0)
        sp.Rational(1, 2) + sp.I / sp.sqrt(12),      # (1/2, 1/√12)
        1 + sp.I / sp.sqrt(3),    # (1, 1/√3)
        sp.Rational(3, 2) + sp.I * sp.sqrt(3) / 2,   # (3/2, √3/2)
        sp.Rational(7, 6) + sp.I * sp.sqrt(11) / 6,  # (7/6, √11/6)
        sp.Rational(1, 6) + sp.I * (sp.sqrt(12) - sp.sqrt(11)) / 6,     # (1/6, (√12 - √11)/6)
        sp.Rational(5, 6) + sp.I * (sp.sqrt(12) - sp.sqrt(11)) / 6,     # (5/6, (√12 - √11)/6)
        sp.Rational(2, 3) + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 6,      # (2/3, (√11 - √3)/6)
        sp.Rational(2, 3) + sp.I * (3 * sp.sqrt(3) - sp.sqrt(11)) / 6,  # (2/3, (3√3 - √11)/6)
        sp.sqrt(33) / 6 + sp.I / sp.sqrt(12),              # (√33/6, 1/√12)
        (sp.sqrt(33) + 3) / 6 + sp.I / sp.sqrt(3),         # ((√33 + 3)/6, 1/√3)
        (sp.sqrt(33) + 1) / 6 + sp.I * (3 * sp.sqrt(3) - sp.sqrt(11)) / 6,    # ((√33 + 1)/6, (3√3 - √11)/6)
        (sp.sqrt(33) - 1) / 6 + sp.I * (3 * sp.sqrt(3) - sp.sqrt(11)) / 6,    # ((√33 - 1)/6, (3√3 - √11)/6)
        (sp.sqrt(33) + 1) / 6 + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 6,        # ((√33 + 1)/6, (√11 - √3)/6)
        (sp.sqrt(33) - 1) / 6 + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 6,        # ((√33 - 1)/6, (√11 - √3)/6)
        (sp.sqrt(33) - 2) / 6 + sp.I * (2 * sp.sqrt(3) - sp.sqrt(11)) / 6,    # ((√33 - 2)/6, (2√3 - √11)/6)
        (sp.sqrt(33) - 4) / 6 + sp.I * (2 * sp.sqrt(3) - sp.sqrt(11)) / 6,    # ((√33 - 4)/6, (2√3 - √11)/6)
        (sp.sqrt(33) + 13) / 12 + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 12,     # ((√33 + 13)/12, (√11 - √3)/12)
        (sp.sqrt(33) + 11) / 12 + sp.I * (sp.sqrt(3) + sp.sqrt(11)) / 12,     # ((√33 + 11)/12, (√3 + √11)/12)
        (sp.sqrt(33) + 9) / 12 + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 4,       # ((√33 + 9)/12, (√11 - √3)/4)
        (sp.sqrt(33) + 9) / 12 + sp.I * (3 * sp.sqrt(3) + sp.sqrt(11)) / 12,  # ((√33 + 9)/12, (3√3 + √11)/12)
        (sp.sqrt(33) + 7) / 12 + sp.I * (sp.sqrt(3) + sp.sqrt(11)) / 12,      # ((√33 + 7)/12, (√3 + √11)/12)
        (sp.sqrt(33) + 7) / 12 + sp.I * (3 * sp.sqrt(3) - sp.sqrt(11)) / 12,  # ((√33 + 7)/12, (3√3 - √11)/12)
        (sp.sqrt(33) + 5) / 12 + sp.I * (5 * sp.sqrt(3) - sp.sqrt(11)) / 12,  # ((√33 + 5)/12, (5√3 - √11)/12)
        (sp.sqrt(33) + 5) / 12 + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 12,      # ((√33 + 5)/12, (√11 - √3)/12)
        (sp.sqrt(33) + 3) / 12 + sp.I * (3 * sp.sqrt(11) - 5 * sp.sqrt(3)) / 12,  # ((√33 + 3)/12, (3√11 - 5√3)/12)
        (sp.sqrt(33) + 3) / 12 + sp.I * (sp.sqrt(3) + sp.sqrt(11)) / 12,      # ((√33 + 3)/12, (√3 + √11)/12)
        (sp.sqrt(33) + 3) / 12 + sp.I * (3 * sp.sqrt(3) - sp.sqrt(11)) / 12,  # ((√33 + 3)/12, (3√3 - √11)/12)
        (sp.sqrt(33) + 1) / 12 + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 12,      # ((√33 + 1)/12, (√11 - √3)/12)
        (sp.sqrt(33) - 1) / 12 + sp.I * (3 * sp.sqrt(3) - sp.sqrt(11)) / 12,  # ((√33 - 1)/12, (3√3 - √11)/12)
        (sp.sqrt(33) - 3) / 12 + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 12,      # ((√33 - 3)/12, (√11 - √3)/12)
        (15 - sp.sqrt(33)) / 12 + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 4,      # ((15 - √33)/12, (√11 - √3)/4)
        (15 - sp.sqrt(33)) / 12 + sp.I * (7 * sp.sqrt(3) - 3 * sp.sqrt(11)) / 12,  # ((15 - √33)/12, (7√3 - 3√11)/12)
        (13 - sp.sqrt(33)) / 12 + sp.I * (3 * sp.sqrt(3) - sp.sqrt(11)) / 12,      # ((13 - √33)/12, (3√3 - √11)/12)
        (11 - sp.sqrt(33)) / 12 + sp.I * (sp.sqrt(11) - sp.sqrt(3)) / 12,     # ((11 - √33)/12, (√11 - √3)/12)
    ]
    # Let Sa be the unit-distance graph whose vertices consist of all points
    # obtained by rotating the points in S around the origin by multiples of 60 degrees
    # and/or by negating their y-coordinates.
    Sa = S.copy()
    angles = [k * sp.pi / 3 for k in range(1, 6)]
    for angle in angles:
        rotated = rotate_points(S, 0, angle)
        Sa = merge_points(Sa, rotated)
    reflected = [sp.conjugate(p) for p in Sa]
    Sa = merge_points(Sa, reflected)
    # Sb is Sa rotated anticlockwise about the origin by 2arcsin(1/4)
    Sb = rotate_points(Sa, 0, 2 * sp.asin(sp.S(1) / 4))
    print("Sa and Sb completed!")
    # Let Y be the union of Sa and Sb with the vertices (1/3, 0) and (−1/3, 0) deleted.
    Y = merge_points(Sa, Sb)
    Y.remove(sp.Rational(1, 3))
    Y.remove(-sp.Rational(1, 3))
    # Rotate Y anticlockwise about (-2,0) by π/2 + arcsin(1/8) to give Ya.
    Ya = rotate_points(Y, -2, sp.pi / 2 + sp.asin(sp.S(1) / 8))
    # Rotate Y anticlockwise about (-2,0) by π/2 − arcsin(1/8) to give Yb.
    Yb = rotate_points(Y, -2, sp.pi / 2 - sp.asin(sp.S(1) / 8))
    print("Ya and Yb completed!")
    # Let G be the union of Ya and Yb.
    G = merge_points(Ya, Yb)
    return G

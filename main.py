import argparse
import networkx as nx
import matplotlib.pyplot as plt
from pysat.solvers import MapleChrono
from generate_points import *
from utils import *


# ==================== Graph Class ====================
class UnitDistanceGraph:
    def __init__(self, points):
        """Create unit distance graph from exact points"""
        self.points = [sp.simplify(p) for p in points]
        self.graph = nx.Graph()
        # Add nodes
        for idx, point in enumerate(self.points):
            self.graph.add_node(idx)
            coord = point_to_coordinate(point)
            self.graph.nodes[idx]['pos'] = (coord[0], coord[1])
        # Build edges (points at distance exactly 1)
        self._build_edges()

    def _build_edges(self):
        """Build edges between points exactly distance 1 apart"""
        n = len(self.points)
        for i in range(n):
            for j in range(i + 1, n):
                dist_approx = abs(complex(self.points[i]) - complex(self.points[j]))
                if abs(dist_approx - 1) < 1e-5:
                    dist_exact = sp.simplify(sp.Abs(self.points[i] - self.points[j]) - 1)
                    if dist_exact == 0:
                        self.graph.add_edge(i, j)

    def verify_all_edges(self):
        """Verify all edges are exactly unit distance"""
        for node_a, node_b in self.graph.edges():
            exact_a = self.points[node_a]
            exact_b = self.points[node_b]
            dist = distance(exact_a, exact_b)
            if sp.simplify(dist - 1) != 0:
                print(f"Error: Edge ({node_a}, {node_b}) is not unit distance")
                print(f"  Point A: {exact_a}")
                print(f"  Point B: {exact_b}")
                print(f"  Distance: {distance}")
                return False
        return True

    def visualize(self, show_labels=False, node_size=30):
        """Visualize the graph"""
        positions = nx.get_node_attributes(self.graph, 'pos')
        plt.figure(figsize=(10, 10))
        nx.draw(
            self.graph, positions,
            node_size=node_size,
            with_labels=show_labels,
            font_size=8,
            edge_color='gray',
            node_color='lightblue'
        )
        plt.title(f"Unit Distance Graph: {self.node_count} vertices, {self.edge_count} edges")
        plt.axis('equal')
        plt.show()

    @property
    def node_count(self):
        return self.graph.number_of_nodes()

    @property
    def edge_count(self):
        return self.graph.number_of_edges()


class GraphFromEdges:
    def __init__(self, edges):
        self.edges = edges
        points = set()
        for i, j in edges:
            points.add(i)
            points.add(j)
        node_count = len(points)
        self.graph = nx.Graph()
        for i in range(node_count):
            self.graph.add_node(i)
        for i, j in edges:
            self.graph.add_edge(i, j)

    @property
    def node_count(self):
        return self.graph.number_of_nodes()

    @property
    def edge_count(self):
        return self.graph.number_of_edges()


# ==================== SAT Solver ====================
class GraphColoringSAT:
    def __init__(self, num_colors=4):
        self.num_colors = num_colors

    def encode_graph(self, graph):
        """Encode graph coloring problem as SAT clauses"""
        clauses = []
        node_to_clauses = {node: [] for node in graph.graph.nodes()}
        clause_to_nodes = []

        def add_clause(_clause, involved_nodes):
            clause_id = len(clauses)
            clauses.append(_clause)
            clause_to_nodes.append(involved_nodes)
            for node in involved_nodes:
                node_to_clauses[node].append(clause_id)

        def var_id(node, color):
            return node * self.num_colors + color + 1

        # Each node has at least one color
        for node in graph.graph.nodes():
            color_vars = [var_id(node, color) for color in range(self.num_colors)]
            add_clause(color_vars, [node])
        # Each node has at most one color
        for node in graph.graph.nodes():
            for color1 in range(self.num_colors):
                for color2 in range(color1 + 1, self.num_colors):
                    clause = [-var_id(node, color1), -var_id(node, color2)]
                    add_clause(clause, [node])
        # Adjacent nodes cannot have same color
        for node_a, node_b in graph.graph.edges():
            for color in range(self.num_colors):
                clause = [-var_id(node_a, color), -var_id(node_b, color)]
                add_clause(clause, [node_a, node_b])
        return clauses, node_to_clauses, clause_to_nodes

    def check_colorability(self, graph, extra_clauses=None):
        """Check if graph is colorable with given number of colors"""
        clauses, _, _ = self.encode_graph(graph)
        if extra_clauses:
            clauses.extend(extra_clauses)
        solver = MapleChrono()
        for clause in clauses:
            solver.add_clause(clause)
        return solver.solve()


# ==================== File Loading ====================
def load_points_from_file(filename):
    """
    Load point coordinates from .txt file
    Format: { real , imag } with SymPy-style sqrt() notation
    """
    points = []
    with open(filename) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            # Remove curly braces
            line = line[1:-1].strip()
            # Split on the comma
            real_str, imag_str = line.split(',', 1)
            real_str = real_str.strip()
            imag_str = imag_str.strip()
            # Parse as sympy expressions
            real_expr = sp.sympify(real_str)
            imag_expr = sp.sympify(imag_str)
            # Create complex number
            point = real_expr + sp.I * imag_expr
            simplified_point = sp.simplify(point)
            points.append(simplified_point)
    return points


def load_edges_from_file(filename):
    edges = []
    with open(filename) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            i, j = line.split()
            i, j = int(i), int(j)
            edges.append((i, j))
    return edges


# ==================== Graph Generation ====================
def generate_graph(graph_type, output_file=None):
    if graph_type == "triangle":
        points = triangle()
    elif graph_type == "h":
        points = create_H()
    elif graph_type == "j":
        points = create_J()
    elif graph_type == "k":
        points = create_K()
    elif graph_type == "l":
        points = create_L()
    elif graph_type == "m":
        points = create_M()
    elif graph_type == "n":  # You can you up!
        points = create_N()
    elif graph_type == "g":
        points = create_G()
    else:
        print(f"Graph type '{graph_type}' not yet implemented")
        return []
    if output_file:
        with open(output_file, 'w') as f:
            for point in points:
                real = sp.simplify(sp.re(point))
                imag = sp.simplify(sp.im(point))
                real_str = str(real)
                imag_str = str(imag)
                f.write(f"{{{real_str}, {imag_str}}}\n")
    return points


# ==================== Main Analysis ====================
def analyze_graph(graph, num_colors, verify_edges, visualize):
    """Main analysis function: create graph and check colorability"""
    # Create unit distance graph
    print(f"Graph created: {graph.node_count} vertices, {graph.edge_count} edges")

    # Verify edges if requested
    if verify_edges:
        print("Verifying all edges are unit distance...")
        if graph.verify_all_edges():
            print("✓ All edges are exactly unit distance")
        else:
            print("✗ Found non-unit distance edges")
            return

    # Create SAT solver
    sat_solver = GraphColoringSAT(num_colors=num_colors)
    # Check colorability
    print(f"Checking if graph is {num_colors}-colorable...")
    is_colorable = sat_solver.check_colorability(graph)
    if is_colorable:
        print(f"✓ Graph IS {num_colors}-colorable")
    else:
        print(f"✗ Graph is NOT {num_colors}-colorable")

    # Visualize if requested
    if visualize:
        print("Visualizing graph...")
        show_labels = (graph.node_count <= 50)
        graph.visualize(show_labels=show_labels)

    # Print summary
    print("=" * 60)
    print("Analysis Summary:")
    print(f"  Vertices: {graph.node_count}")
    print(f"  Edges: {graph.edge_count}")
    print(f"  {num_colors}-colorable: {'Yes' if is_colorable else 'No'}")
    print("=" * 60)


# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(description="Analyze unit distance graphs for the Hadwiger-Nelson problem")
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-points", type=str, help="Load points from .txt file")
    input_group.add_argument("--input-edges", type=str, help="Load edges from .txt file")
    input_group.add_argument("--generate", type=str, help="Generate a predefined graph")
    # Analysis options
    parser.add_argument("--colors", type=int, default=4, help="Number of colors to test (default: 4)")
    parser.add_argument("--verify", action="store_true", help="Verify all edges are unit distance")
    parser.add_argument("--visualize", action="store_true", help="Visualize the graph")
    # Output option for generated graphs
    parser.add_argument("--output-file", type=str, help="Save generated graph to file")
    args = parser.parse_args()

    # Load or generate points/edges
    graph = None
    if args.input_points:
        points = load_points_from_file(args.input_points)
        print(f"Loaded {len(points)} points from {args.input_points}")
        graph = UnitDistanceGraph(points)
    elif args.input_edges:
        edges = load_edges_from_file(args.input_edges)
        print(f"Loaded {len(edges)} edges from {args.input_edges}")
        graph = GraphFromEdges(edges)
    elif args.generate:
        points = generate_graph(args.generate, args.output_edges)
        print(f"Generated {len(points)} points for graph type '{args.generate}'")
        graph = UnitDistanceGraph(points)
    print(f"Graph created: {graph.node_count} vertices, {graph.edge_count} edges")

    # Analyze the graph
    analyze_graph(graph, args.colors, args.verify, args.visualize)


if __name__ == "__main__":
    main()

# python main.py --input-edges best_train_graph_chromatic4_20251221_201123_edges.txt --colors 4
# python main.py --input-points l_graph.txt --colors 4 --visualize
# python main.py --generate triangle --colors 3 --visualize
# python main.py --generate g --colors 4 --output-file g_graph.txt --visualize
# python main.py --generate m --colors 4 --output-file m_graph.txt --visualize

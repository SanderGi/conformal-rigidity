import networkx as nx
from scipy.linalg import eigh


def nx_to_adj_list(G):
    """Convert a NetworkX graph to an adjacency list."""
    adj_list = []
    for v in G.nodes():
        adj_list.append([u for u in G.neighbors(v)])
    return adj_list


def laplacian_matrix(G):
    """Return the Laplacian matrix of graph G as a NumPy array."""
    return nx.laplacian_matrix(G).toarray()


def compute_eigenvalues(L):
    """Compute eigenvalues (sorted in increasing order) of Laplacian L."""
    # eigh returns sorted eigenvalues for symmetric matrices.
    eigvals, eigvecs = eigh(L)
    return eigvals, eigvecs


def load_graph(filename):
    """Load an nx graph from a file."""
    with open(filename) as f:
        lines = f.readlines()
        n, e = map(int, lines.pop(0).split())
        adj_list = [tuple(map(int, line.split())) for line in lines]
        min_vertex = min([min(edge) for edge in adj_list])
        adj_list = [tuple(v - min_vertex for v in edge) for edge in adj_list]
        edges = []
        for v, neighbors in enumerate(adj_list):
            for n in neighbors:
                edges.append((v, n))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        return G

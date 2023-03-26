import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools as it

# return a graph from the Hamiltonian matrix of a graph. Remove the diagonal elements and set the rest to 1.
def get_graph_from_h(h):
    g = nx.from_numpy_array(h)
    g.remove_edges_from(nx.selfloop_edges(g))
    for i, j in g.edges():
        g[i][j]['weight'] = 1
    return g

# return a graph of n nodes. Nodes are integers, edges are tuples of integers. One node is connected to all other nodes. The other nodes are connected to each other.

def get_graph(n):
    graph = nx.Graph()
    graph.add_nodes_from(range(1, n))
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j)
    return graph


# return all paths of a certain length in a graph, with a given start and end node. Allow node repitition.
def get_paths_edges(G, source, target, length, path=None, edges=None):
    if path is None:
        path = [source]
        edges = []

    if len(path) == length + 1:
        if path[-1] == target:
            yield path, edges
        return

    for neighbor in G.neighbors(path[-1]):
        edge = (path[-1], neighbor)
        yield from get_paths_edges(G, source, target, length, path + [neighbor], edges + [edge])

# return a path that only contain the source and target once in a graph, with a given start and end node. Allow node repitition.


def get_order2_paths_edges(G, source, target, length, path=None):
    paths_edges = get_paths_edges(G, source, target, length, path)
    for path, edge in paths_edges:
        p_mid = path[1:-1]
        if p_mid.count(source) == 0 and p_mid.count(target) == 0:
            yield path, edge

def is_path_reverse(path, paths_set):
    reversed_path = tuple(reversed(path))
    return reversed_path in paths_set

def remove_reverse_paths(paths_and_edges):
    unique_paths_set = set()
    filtered_paths_and_edges = []

    for path, edges in paths_and_edges:
        path_tuple = tuple(path)
        if not is_path_reverse(path_tuple, unique_paths_set):
            unique_paths_set.add(path_tuple)
            filtered_paths_and_edges.append((path, edges))

    return filtered_paths_and_edges

def gen_rate_order(h: np.ndarray, kbT, w, s, t_max, order, nitn=10, neval=1000):
    g = get_graph_from_h(h)
    paths_and_edges = list(get_paths_edges(g, 0, 0, order+2))
    print("Number of paths: ", len(paths_and_edges))
    k = 0
    for path_i, edges_i in paths_and_edges:
        d = gen_rate_edge(h, edges_i, kbT, w, s, t_max, nitn, neval)
        k += d
        print("path: ", path_i, "rate correction: ", d)
    return k

def gen_rate_edge(h: np.ndarray, edges, kbT, w, s, t_max, nitn=10, neval=1000):
    e = np.diagonal(h)
    s = np.array(s)
    w = np.array(w)
    order = len(edges) - 2
    w_sq = w ** 2

    s = s
    E = e

    sub_list = edges
    # sub_list = {0: ("D", "A1")}
    # for i in range(1, order + 1):
    #     if i % 2 == 1:
    #         sub_list[i] = ("A1", "A2")
    #     else:
    #         sub_list[i] = ("A2", "A1")
    # if order % 2 == 0:
    #     sub_list[order + 1] = ("A1", "D")
    # if order % 2 == 1:
    #     sub_list[order + 1] = ("A2", "D")

    delta = {}
    for i in range(order + 2):
        l, r = sub_list[i]
        delta[i] = s[l] - s[r]

    coth = 1 / np.tanh(w / (2 * kbT))
    const_exponent = np.sum(-coth * [delta[i] ** 2 for i in range(order + 2)], axis=0) / (2 * w_sq * np.pi)

    # Generate exponent
    def exponent(*t):
        """
        Args:
            t : a list storing time variables. E.g., for order 3, the list t has three elements

        Returns:
            float
        """
        pre = {}
        summand = 0
        for m, n in it.combinations(range(len(t)), 2):
            summand += delta[m] * delta[n] / w_sq / np.pi \
                       * (- coth * np.cos(w * (t[m] - t[n]))
                          + 1j * np.sin(w * (t[m] - t[n]))
                          )

        return np.sum(summand + const_exponent)

    def time_factor(*t):
        f = 1
        for i in range(len(t)):
            k, l = sub_list[i]
            f *= np.exp(1j * t[i] * (E[k] - E[l]))
        return f

    # changing variables

    def y2t(y, beta):
        t = []
        for i, yi in enumerate(y):
            t.append(np.prod(y[:i + 1]) / beta ** i)
        return t

    def t2y_jacobian(y, beta):
        jacobian = 1
        n = len(y)
        for i, yi in enumerate(y[:-1]):
            jacobian *= (yi / beta) ** (n - 1 - i)
        return jacobian

    def integrand(y):
        """

        Args: y (): y_ is the list of y1, y2, ..., y_{n-1} for the n-th order. Note the argument of the functions
        time_factor() and exponent() is t0, t_1, t_2, ..., t_{n-1}.

        Returns: float

        """
        t_ = y2t(y, t_max)  # t1, t2, ..., t_{n-1}
        return np.real(
            (-1j) ** (order + 2)
            * time_factor(t_max, *t_)
            * np.exp(exponent(t_max, *t_))
            * t2y_jacobian(y, t_max)
        )

    import vegas

    int_interval = [0, t_max]
    integrator = vegas.Integrator([int_interval] * (order + 1))

    integral = integrator(integrand, nitn=nitn, neval=neval).mean
    coupling_factor = 1
    for edge in edges:
        l, r = edge
        coupling_factor *= h[*edge]
    return -1 * coupling_factor * integral


if __name__ == "__main__":
    order = 5
    h = np.array([[1.5, 2, 1], [1.1, 0.1, 1], [1, 1, 0.9]], dtype=float)
    g = get_graph_from_h(h)
    p = get_order2_paths_edges(g, 0, 0, order)
    p = remove_reverse_paths(p)
    for i, pi in enumerate(p):
        print(i, pi)
    g = get_graph(3)
    p = get_order2_paths_edges(g, 0, 0, order)
    for i, pi in enumerate(p):
        print(i, pi)
    nx.draw(g)
    plt.savefig("graph.png")

import matplotlib.pyplot as plt
import networkx as nx
# return all paths of a certain length in a graph, with a given start and end node. Allow node repitition.
def get_paths(G, source, target, length, path=None):
    if path is None:
        path = [source]

    if len(path) == length + 1:
        if path[-1] == target:
            yield path
        return

    for neighbor in G.neighbors(path[-1]):
        yield from get_paths(G, source, target, length, path + [neighbor])

# return a graph of n nodes. Nodes are integers, edges are tuples of integers. One node is connected to all other nodes. The other nodes are connected to each other.
def get_graph(n):
    graph = nx.Graph()
    graph.add_nodes_from(range(1,n))
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j)
    return graph



g = get_graph(3)

p = get_paths(g, 0, 0, 4)
print([*p])
nx.draw(g)
plt.savefig("graph.png")
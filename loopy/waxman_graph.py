# adapted from networkx waxman_graph implementation
import random
import math

def directed_waxman_graph(n, alpha=0.4, beta=0.1, domain_scale=1.0, dimensionality=3):
    r"""Return a directed Waxman random graph.

    The Waxman random graph model places ``n`` nodes uniformly at random in a
    hypercube domain. Each pair of nodes at Euclidean distance `d` is joined
    by an edge with probability

    .. math::
            p = \alpha \exp(-d / \beta L).

    L is the maximum distance between any pair of nodes.

    Parameters
    ----------
    n : int
        Number of nodes
    alpha: float
        Model parameter; similar to mean
    beta: float
        Model parameter; similar to variance
    domain_scale : float
        Domain scale, which scales the hypercube domain
    dimensionality : int
        Dimensionality of the hypercube domain

    Returns
    -------
    G: DiGraph

    References
    ----------
    .. [1]  B. M. Waxman, Routing of multipoint connections.
       IEEE J. Select. Areas Commun. 6(9),(1988) 1617-1622.
    """
    # build graph of n nodes with random positions in a hypercube
    G = dict([(i, set()) for i in range(n)])

    position_map = {}
    for n in G:
        position = tuple([random.random() * domain_scale for i in range(dimensionality)])
        position_map[n] = position

    def euclidean_distance(p1, p2):
        return math.sqrt(sum([(c1-c2)**2 for c1, c2 in zip(p1, p2)]))

    l = 0
    positions = list(position_map.values())
    while positions:
        position1 = positions.pop()
        for position2 in positions:
            distance = euclidean_distance(position1, position2)
            if distance > l:
                l = distance

    nodes = list(G.keys())
    # Waxman-1 model
    # try all pairs, connect randomly based on euclidean distance
    while nodes:
        u = nodes.pop()
        position1 = position_map[u]
        for v in nodes:
            position2 = position_map[v]
            # r = euclidean_distance(position1, position2)
            r = math.sqrt(sum([(c1-c2)**2 for c1, c2 in zip(position1, position2)]))
            # premature optimization
            if random.random() < alpha*math.exp(-r/(beta*l)):
                if random.random() < 0.5:
                    G[u].add(v)
                else:
                    G[v].add(u)
    return G

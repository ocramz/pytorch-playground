class Graph:
    def __init__(self):
        self.verticesDict = {}
        self.edgesDict = {}
    def __repr__(self):
        return f'Nodes : {str(self.verticesDict)}, edges : {str(self.edgesDict)}'
    def numNodes(self):
        return len(self.verticesDict)
    def numEdges(self):
        return len(self.edgesDict)
    def nodes(self):
        """stream node indices and resp. embedding"""
        for n in self.verticesDict.items():
            yield n
    def edges(self):
        """stream edge index tuples"""
        for e in self.edgesDict.items():
            yield e
    def lookupNode(self, k):
        try:
            v0 = self.verticesDict[k]
            return v0
        except KeyError as e:
            return None
    def lookupEdge(self, i):
        try:
            i2 = self.edgesDict[i]
            return i2
        except KeyError as e:
            return None
    def node(self, i, vv):
        """add a node, overwriting any previous value"""
        self.verticesDict[i] = vv
    def edge(self, i1, i2):
        """add an edge"""
        i2m = self.lookupEdge(i1)
        if i2m is None:
            self.edgesDict[i1] = [i2]
        else:
            self.edgesDict[i1] = [i2] + i2m
    def biEdge(self, i1, i2):
        """add bidirectional edge"""
        self.edge(i1, i2)
        self.edge(i2, i1)
    def neighbors(self, refIx: int, transpose=False):
        """return iterator of neighbors of a node
        :param transpose: return neighbors in transposed graph
        """
        if transpose:
            for i, ns in self.edgesDict.items():
                if refIx in ns:
                    yield i
        else:
            for i, ns in self.edgesDict.items():
                if i == refIx:
                    for n in ns:
                        yield n

def graphFromList(ll):
    g = Graph()
    for i1, i2, v1, v2 in ll:
        g.node(i1, v1)
        g.node(i2, v2)
        g.edge(i1, i2)
    return g

def unGraphFromList(ll):
    g = Graph()
    for i1, i2, v1, v2 in ll:
        g.node(i1, v1)
        g.node(i2, v2)
        g.biEdge(i1, i2)
    return g

## test data
g0 = graphFromList([(1, 1, 'x', 'z'), (1, 2, 'y', 'w'), (3, 2, 'a', 'b'), (4, 3, 'u', 'v'), (5, 3, 'x', 'y')])

g1 = unGraphFromList([(1, 2, 'y', 'w'), (3, 2, 'a', 'b'), (4, 3, 'u', 'v'), (5, 3, 'x', 'y')])
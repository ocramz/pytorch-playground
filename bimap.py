
class IMap:
    def __init__(self):
        self.i = 0
        self.d = {}
    def __len__(self):
        return len(self.d)
    def __repr__(self):
        return f'{str(self.d)}'
    def __iter__(self):
        for e in self.d.items():
            yield e
    def values(self):
        for k, v in self.__iter__():
            yield v
    def lookup(self, k):
        try:
            v = self.d[k]
            return v
        except KeyError as e:
            return None
    def insert(self, k, v):
        v1 = self.lookup(k)
        if v1 is None:
            self.d[k] = v
            self.i += 1
    def append(self, v):
        k = self.i
        self.d[k] = v
        self.i += 1
    def fromIter(self, vs):
        for v in vs:
            self.append(v)

class BiMap:
    """bidirectional map"""
    def __init__(self, xs=[], defaultIx=0):
        self.i = 0
        self.dk = IMap()
        self.dv = IMap()
        self.defaultIx = defaultIx
        self.fromIter(xs)
    def __len__(self):
        return len(self.dk)
    def __repr__(self):
        return f'{str(self.dk)}'
    def __iter__(self):
        return iter(self.dk.d.items())
    def lookupK(self, k):
        """lookup a key"""
        return self.dk.lookup(k)
    def lookupV(self, v):
        """lookup a value"""
        return self.dv.lookup(v)
    def lookupVD(self, v):
        """lookup a value with default in case not found"""
        km = self.lookupV(v)
        if km is None:
            return self.defaultIx
        else:
            return km
    def lookupKs(self, ks):
        """lookup multiple keys"""
        for k in ks:
            yield self.lookupK(k)
    def lookupVs(self, vs):
        """lookup multiple values"""
        for v in vs:
            yield self.lookupV(v)
    def lookupVDs(self, vs):
        for v in vs:
            yield self.lookupVD(v)
    def insert(self, k, v):
        """insert with duplication"""
        vl = self.lookupK(k)
        if vl is None:
            self.dk.insert(k, v)
            self.dv.insert(v, k)
    def insertV(self, v):
        """insert without duplication"""
        kl = self.lookupV(v)
        if kl is None:
            k = self.i
            self.dk.insert(k, v)
            self.dv.insert(v, k)
            self.i += 1
    def append(self, v):
        k = self.i
        self.dk.insert(k, v)
        self.dv.insert(v, k)
        self.i += 1
    def fromIter(self, vs):
        """append elements without duplication"""
        for v in vs:
            self.insertV(v)

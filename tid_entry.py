class TIDEntry:
    def __init__(self, x, Ix):
        self.x = x
        self.Ix = Ix
        self.weight = len(Ix)

    def __lt__(self, other):
        return self.weight < other.weight

    def __gt__(self, other):
        return self.weight > other.weight

    def __eq__(self, other):
        return self.weight == other.weight

from typing import Any


class Edge:
    def __init__(self, p1, p2):
        # Convert to tuples and sort so that ordering doesn't matter
        tp1 = tuple(p1) if hasattr(p1, '__iter__') and not isinstance(p1, str) else p1
        tp2 = tuple(p2) if hasattr(p2, '__iter__') and not isinstance(p2, str) else p2
        # Ensure p1 <= p2 for consistent hashing
        if tp1 <= tp2:
            self.p1, self.p2 = tp1, tp2
        else:
            self.p1, self.p2 = tp2, tp1

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.p1 == other.p1 and self.p2 == other.p2
    
    def __hash__(self):
        return hash((self.p1, self.p2))
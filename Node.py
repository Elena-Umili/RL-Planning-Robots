class Node:

    def __init__(self, x, v, a, c=1):
        self.x = x
        self.v = v
        self.a = a
        self.c = c
        self.sons = []

    def expand(self, x1, v1, a1, c1=1):
        self.sons.append(Node(x1, v1, a1, c1))

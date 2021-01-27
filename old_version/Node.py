class Node:

    def __init__(self, x, v, a, c=1):
        self.x = x
        self.v = v
        #a è l'azione con cui ci sono arrivato al nodo
        self.a = a
        self.c = c
        self.sons = []

    def expand(self, x1, v1, a1, c1=1):
        self.sons.append(Node(x1, v1, a1, c1))

    def info2string(self):
        return "[x={}, v={}, a={}, c={}]".format(self.x, self.v, self.a, self.c)

    def print_parentetic(self):
        print("( {} : [".format(self.info2string()) )
        for i in range(len(self.sons)):
            self.sons[i].print_parentetic()
        print("] )")

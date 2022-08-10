from itertools import zip_longest

class Node:
    def __init__(self, left=None, right=None, idx=None, is_leaf=False) -> None:
        self.left = left if not is_leaf else None
        self.right = right if not is_leaf else None
        self.parent = None
        self.is_leaf = is_leaf
        self.value = 0 if is_leaf else self.left.value + self.right.value
        self.idx = idx
        if self.left is not None:
            self.left.parent = self
        if self.right is not None:
            self.right.parent = self

    def update_value(self, value):
        old_value = self.value
        if self.is_leaf:
            self.value = value
            self.parent.update_value(self.value - old_value)
        else:
            self.value += value
            if self.parent is not None:
                self.parent.update_value(self.value - old_value)

class SumTree:
    def __init__(self, capacity):
        self.leaf_nodes = [Node(idx=i, is_leaf=True) for i in range(capacity)]
        all_nodes = self.leaf_nodes

        while len(all_nodes) > 1:
            all_nodes = [Node(*pair) for pair in zip_longest(all_nodes[::2],all_nodes[1::2], fillvalue=None)]

        self.head = all_nodes[0]

    def retrieve_node(self, value, node=None):
        node = node if node is not None else self.head

        if node.is_leaf:
            return node
        
        if node.left.value >= value:
            return self.retrieve_node(value, node.left)
        else:
            return self.retrieve_node(value - node.left.value, node.right)


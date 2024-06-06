import json
import math
from random import randint, random
from statistics import mean
import torch


MIN_DEPTH = 2  # minimal initial random tree depth
MAX_DEPTH = 4  # maximal initial random tree depth

def add(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        x, y = torch.broadcast_tensors(x, y)
        return x + y
    else:
        return x + y


def sub(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        x, y = torch.broadcast_tensors(x, y)
        return x - y
    else:
        return x - y


def mul(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        x, y = torch.broadcast_tensors(x, y)
        return x * y
    else:
        return x * y


def div(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        x, y = torch.broadcast_tensors(x, y)
        return x / torch.norm(y)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x / abs(y)
    else:
        raise TypeError('Input types not supported')


def sqr(x):
    if isinstance(x, torch.Tensor):
        return x * x
    else:
        return x * x


def neg(x):
    if isinstance(x, torch.Tensor):
        return -x
    else:
        return -x


def abs(x):
    if isinstance(x, torch.Tensor):
        return torch.abs(x)
    else:
        return math.fabs(x)


def log(x):
    if isinstance(x, torch.Tensor):
        return torch.log(torch.abs(x) + 0.001)
    else:
        return math.log(abs(x) + 0.001)


def sqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(torch.abs(x))
    else:
        return math.sqrt(abs(x))


def tanh(x):
    if isinstance(x, torch.Tensor):
        return torch.tanh(x)
    else:
        return math.tanh(x)


def pow(x):
    if isinstance(x, torch.Tensor):
        return torch.pow(x, 2)
    else:
        return x**2


def skp(x):
    return x


def mms(x):
    # min-max scale to [0, 1]
    if isinstance(x, torch.Tensor):
        return (x - x.min()) / (x.max() - x.min())
    else:
        return (x - min(x)) / (max(x) - min(x))


def zsn(x):
    # z-score normalization
    if isinstance(x, torch.Tensor):
        return (x - x.mean()) / x.std()
    else:
        return (x - mean(x)) / x.std()


def exp(x):
    if isinstance(x, torch.Tensor):
        return torch.exp(x.clamp(max=100))  # Clamp to prevent overflow
    else:
        return math.exp(min(x, 100))  # Use Python's math.exp for scalars


UNARY_FUNCTIONS = [sqr, neg, abs, log, exp, sqrt, tanh, pow, skp, mms, zsn]
BINARY_FUNCTIONS = [add, sub, mul, div]

FUNCTIONS = UNARY_FUNCTIONS + BINARY_FUNCTIONS
TERMINALS = ['W', 'G', 'X']


class GPTree:
    """
    Represents a Genetic Programming tree.

    Attributes:
        data: The function or terminal symbol at the root of the tree.
        left: The left subtree.
        right: The right subtree.

    Methods:
        save_tree(filename): Saves the tree to a file in JSON format.
        _serialize_tree(): Serializes the tree into a dictionary.
        load_tree(filename): Loads a tree from a file in JSON format.
        _deserialize_tree(data): Deserializes a tree from a dictionary.
        _get_function_from_label(label): Retrieves the function from its label.
        node_label(): Returns the string label of the node.
        draw(dot, count): Draws the tree using Graphviz.
        draw_tree(fname, footer): Draws the tree and saves it as an image.
        compute_tree(W, G, X=None): Computes the value of the tree given input variables.
        forward(W, G, X=None): Alias for compute_tree.
        random_tree(grow, max_depth, depth=0): Generates a random tree using the grow or full method.
        mutation(): Performs mutation on the tree.
        size(): Returns the size of the tree in nodes.
        build_subtree(): Builds a subtree rooted at the current node.
        scan_tree(count, second): Scans the tree and returns a subtree at a given position.
        crossover(other): Performs crossover with another tree.
        tree_to_string(node, op_symbols=FUNCTIONS, terminal_symbols=TERMINALS): Converts the tree to a string representation.
        string_to_tree(s, op_symbols=FUNCTIONS, terminal_symbols=TERMINALS): Converts a string representation to a tree.
    """

    def __init__(self, data=None, left=None, right=None):
        self.data = data  # function or terminal
        self.left = left
        self.right = right

    def save_tree(self, filename):
        tree_data = self._serialize_tree()
        with open(filename, 'w') as file:
            json.dump(tree_data, file, indent=4)

    def _serialize_tree(self):
        data = {'data': self.node_label()}
        if self.left:
            data['left'] = self.left._serialize_tree()
        if self.right:
            data['right'] = self.right._serialize_tree()
        return data

    def __repr__(self):
        return self.tree_to_string(self)

    @staticmethod
    def load_tree(filename):
        with open(filename, 'r') as file:
            tree_data = json.load(file)
        return GPTree._deserialize_tree(tree_data)

    @staticmethod
    def _deserialize_tree(data):
        node = GPTree()
        node.data = GPTree._get_function_from_label(data['data'])
        if 'left' in data:
            node.left = GPTree._deserialize_tree(data['left'])
        if 'right' in data:
            node.right = GPTree._deserialize_tree(data['right'])
        return node

    @staticmethod
    def _get_function_from_label(label):
        # Check if the label is a function name
        if label in [f.__name__ for f in FUNCTIONS]:
            return next(f for f in FUNCTIONS if f.__name__ == label)
        # Check if the label is a terminal symbol
        elif label in [str(t)
                       for t in TERMINALS]:  # Convert terminals to strings
            # Convert label back to its original type (int or str)
            if label.isdigit() or (label.startswith('-')
                                   and label[1:].isdigit()):
                return int(label)  # Convert to integer
            return label  # Keep as string
        else:
            raise ValueError(f'Unknown label: {label}')

    def node_label(self): 
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else:
            return str(self.data)

    def compute_tree(self, W, G, X):
        if self.data in FUNCTIONS:
            try:
                if self.data in UNARY_FUNCTIONS:
                    return self.data(self.left.compute_tree(W, G, X))
                else:
                    return self.data(
                        self.left.compute_tree(W, G, X),
                        self.right.compute_tree(W, G, X))
            except Exception as e:
                breakpoint()
        elif self.data == 'W':
            return W
        elif self.data == 'G':
            return G
        elif self.data == 'X':
            return X
        else:
            shape = W.shape if isinstance(W, torch.Tensor) else G.shape
            return torch.full(shape, self.data, dtype=torch.float32)

    def forward(self, W, G, X):
        return self.compute_tree(W, G, X)

    def aggregate_leaf(self):
        leaf_counts = {'W': 0, 'G': 0, 'X': 0}
        self._aggregate_leaf_helper(leaf_counts)
        return leaf_counts

    def _aggregate_leaf_helper(self, leaf_counts):
        if self.data in TERMINALS:
            leaf_counts[str(self.data)] += 1
        if self.left is not None:
            self.left._aggregate_leaf_helper(leaf_counts)
        if self.right is not None:
            self.right._aggregate_leaf_helper(leaf_counts)

    def aggregate_ops(self):
        ops_counts = {func.__name__: 0 for func in FUNCTIONS}
        self._aggregate_ops_helper(ops_counts)
        return ops_counts

    def _aggregate_ops_helper(self, ops_counts):
        if self.data in FUNCTIONS:
            ops_counts[self.data.__name__] += 1
        if self.left is not None:
            self.left._aggregate_ops_helper(ops_counts)
        if self.right is not None:
            self.right._aggregate_ops_helper(ops_counts)

    def random_tree(self, grow, max_depth, depth=0):
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        elif depth >= max_depth:
            self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]

        if self.data in UNARY_FUNCTIONS:  # Unary functions
            self.left = GPTree()
            self.left.random_tree(grow, max_depth, depth=depth + 1)
        elif self.data in FUNCTIONS:  # Binary functions
            self.left = GPTree()
            self.left.random_tree(grow, max_depth, depth=depth + 1)
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth=depth + 1)

    def size(self):  # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):  # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left: t.left = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def check_X_unary(self):
        return self._check_X_unary_helper(parent=None)

    def _check_X_unary_helper(self, parent):
        # Check if current node is 'X' and parent is a unary function
        if self.data == 'X' and parent in UNARY_FUNCTIONS:
            return True

        # Recursively check left and right subtrees if they exist
        left_check = self.left._check_X_unary_helper(
            self.data) if self.left else False
        right_check = self.right._check_X_unary_helper(
            self.data) if self.right else False

        return left_check or right_check

    def scan_tree(self, count, second):
        count[0] -= 1
        if count[0] <= 1:
            if not second:  # return subtree rooted here
                return self.build_subtree()
            else:  # glue subtree here
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1:
                ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1:
                ret = self.right.scan_tree(count, second)
            return ret

    @staticmethod
    def tree_to_string(node, op_symbols=FUNCTIONS, terminal_symbols=TERMINALS):
        if node is None or node.data is None:
            return '#'
        if node.data in op_symbols:
            return f'({GPTree.tree_to_string(node.left)}) {node.node_label()} ({GPTree.tree_to_string(node.right)})'
        elif node.data in terminal_symbols:
            return node.data
        else:
            raise ValueError(f'Unknown data {node.data}')

    @staticmethod
    def string_to_tree(s, op_symbols=FUNCTIONS, terminal_symbols=TERMINALS):

        def find_main_operator(subs):
            balance = 0
            for i, char in enumerate(subs):
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                elif balance == 0 and char in [
                        op.__name__ for op in FUNCTIONS
                ]:
                    return i, subs[i:i + 1]
            return -1, None

        def parse(subs):
            subs = subs.strip()

            if subs == '#':
                return None

            # Handling parentheses and nested expressions
            if subs.startswith('(') and subs.endswith(')'):
                return parse(subs[1:-1])

            op_index, op_name = find_main_operator(subs)
            if op_name:
                if op_name in [op.__name__ for op in UNARY_FUNCTIONS]:
                    left_str = subs[:op_index].strip()
                    left_node = parse(left_str) if left_str != '#' else None
                    return GPTree(
                        GPTree._get_function_from_label(op_name), left_node,
                        None)
                else:
                    left_str = subs[:op_index].strip()
                    right_str = subs[op_index + len(op_name):].strip()
                    left_node = parse(left_str)
                    right_node = parse(right_str)
                    return GPTree(
                        GPTree._get_function_from_label(op_name), left_node,
                        right_node)

            if subs in terminal_symbols:
                return GPTree(subs)

            if subs.isdigit() or (subs.startswith('-') and subs[1:].isdigit()):
                return GPTree(int(subs))

            raise ValueError(f'Unknown substring: {subs}')

        return parse(s)

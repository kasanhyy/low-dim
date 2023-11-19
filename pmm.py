# This is the code of the PMM method
import bokeh.colors.color
import numpy as np


class binaryTreeNode:
    def __init__(self, a, b, l):
        self.smallest = a  # A d-tuple indicating the smallest point in the region
        self.largest = b  # A d-tuple indicating the largest point in the region
        self.lchild = None
        self.rchild = None
        self.level = l  # The level of the node. Root = 0.
        self.count = 0
        self.noisy = 0

    def split(self):
        a = self.smallest
        b = self.largest
        split_pos = np.argmax(b - a)
        a_new = np.copy(a)
        b_new = np.copy(b)
        a_new[split_pos] = (a[split_pos] + b[split_pos]) / 2
        b_new[split_pos] = (a[split_pos] + b[split_pos]) / 2
        self.lchild = binaryTreeNode(a, b_new, self.level + 1)
        self.rchild = binaryTreeNode(a_new, b, self.level + 1)

    def BFS(self, func):
        if self is None:
            return
        queue = [self]
        while queue:
            node = queue.pop(0)
            func(node)
            if node.lchild is not None:
                queue.append(node.lchild)
            if node.rchild is not None:
                queue.append(node.rchild)

    def growTree(self, r):  # Grow the tree from self to level r
        if self.level >= r:
            return
        self.split()
        self.lchild.growTree(r)
        self.rchild.growTree(r)

    def countTrueData(self, data, scale):
        if self.lchild:
            self.lchild.countTrueData(data, scale)
            self.rchild.countTrueData(data, scale)
            self.count = self.lchild.count + self.rchild.count
        else:
            a = self.smallest
            b = self.largest
            b2 = b.copy()
            for i in range(len(b)):
                b2[i] = 1.01 if b[i] == 1 else b[i]

            indicator = (a <= data).all(axis=1) & (data < b2).all(axis=1)
            self.count = np.count_nonzero(indicator) * scale

    def addNoise(self, r, eps=1):
        d = len(self.smallest)
        s = 2 ** (0.5 * (1 - 1 / d) * (r - self.level)) / eps
        noise = np.random.laplace(scale=s)
        if noise >= 0:
            noise = np.floor(noise)
        else:
            noise = np.ceil(noise)
        self.noisy = max(noise + self.count, 0)

    def forceConsistency(self):
        if self.lchild is None:
            return
        children_sum = self.lchild.noisy + self.rchild.noisy
        if children_sum == 0:
            self.lchild.noisy = np.floor(self.noisy / 2)
        else:
            self.lchild.noisy = np.floor(self.noisy * self.lchild.noisy / children_sum)
        self.rchild.noisy = self.noisy - self.lchild.noisy

    # Return synthetic data randomly distributed in the subregion
    def synData(self):
        a = self.smallest
        b = self.largest
        return np.random.rand(int(self.noisy), len(a)) * (b - a) + a


def printCount(item):
    print(item.smallest, item.count, item.noisy)


# To generate data, eps is the privacy parameter
def pmm_data(true_data, eps=1, scale=1):
    """
    :param true_data: original data
    :param eps: privacy parameter
    :param scale: Repetition number if dataset is small
    :return: synthetic dataset generating by PMM
    """
    n, d = true_data.shape
    r = int(np.ceil(np.log2(eps * n)))

    # Generate the tree by DFS
    root = binaryTreeNode(np.zeros(d), np.ones(d), 0)
    root.growTree(r)

    # Count true data in each region in the tree
    # root.BFS(lambda x: x.countTrueData(true_data))
    root.countTrueData(true_data, scale=scale)
    # Add noise to each node in the tree
    root.BFS(lambda x: x.addNoise(r, eps))
    # Force consistency
    root.BFS(lambda x: x.forceConsistency())

    # Generate synthetic data
    queue = [root]
    syn = None
    while queue:
        node = queue.pop(0)
        if node.level < r:
            queue.append(node.lchild)
            queue.append(node.rchild)
        else:
            if syn is None:
                syn = node.synData()
            else:
                syn = np.concatenate((syn, node.synData()))

    return syn

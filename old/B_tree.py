class BTreeKey:
    def __init__(self, val, next):
        self.val = val
        self.next = next

    def __eq__(self, other):
        if self.val == other.val:
            return True
        return False

    def __lt__(self, other):
        if self.val < other.val:
            return True
        return False

    def __gt__(self, other):
        if self.val > other.val:
            return True
        return False

    def __le__(self, other):
        if self.val <= other.val:
            return True
        return False

    def __ge__(self, other):
        if self.val >= other.val:
            return True
        return False

    def __str__(self):
        return f'({self.val}, {self.next})'

    def print(self):
        print(f'({self.val},{self.next})')

class BTreeNode:
    def __init__(self, leaf=False):
        """btree node contains multiple elements"""
        self.leaf = leaf
        self.keys = []  # BTreeKeys
        self.child = [] # BTreeNodes

class BTree():
    def __init__(self, t=2):
        self.root = BTreeNode(leaf=True)
        self.t = t

    def insert(self, k):
        """k - BTreeKey object (points to next key)"""
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.child.insert(0, root)
            self.split_child(temp, 0)
            self.insert_non_full(temp, k)
        else:
            self.insert_non_full(root, k)

    def insert_non_full(self, x, k):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append(None)     # adding element to increase size by 1
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.child[i].keys) == (2 * self.t) - 1:
                self.split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self.insert_non_full(x.child[i], k)

    def split_child(self, x, i):
        t = self.t
        y = x.child[i]
        z = BTreeNode(y.leaf)
        x.child.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        z.keys = y.keys[t: (2 * t) - 1]
        y.keys = y.keys[0: t - 1]
        if not y.leaf:
            buffer = y.child
            z.child = y.child[t:2*t]
            y.child = y.child[0:t] # removes (6,9)

    def search_key(self, k, x=None):
        if x is not None:
            i = 0

            # increment i until k > current element in node
            while k > x.keys[i]:
                i += 1

                # this prevents out of bound access
                if i == len(x.keys):
                    i -= 1
                    break

            # check if element found
            # if not and leaf - element not in tree
            # otherwise seek in child
            if k == x.keys[i]:
                return x.keys[i]
            elif x.leaf:
                return None
            else:
                # search in tree. if fail, return current key
                # which is > k
                res1 = self.search_key(k, x.child[i])
                res2 = self.search_key(k, x.child[i+1])
                if res1 is not None:
                    return res1
                elif res2 is not None:
                    return res2
                else:
                    if x.keys[i] < k:
                        return None

                    return x.keys[i]

        else:
            return self.search_key(k, self.root)

    def print_tree(self, x, l=0):
        if not x:
            x = self.root

        print(f'Level {l}, {len(x.keys)} elements:')
        for i in x.keys:
            print(f'{i} ')
        print()
        l += 1
        if len(x.child) > 0:
            for i in x.child:
                self.print_tree(i, l)

def main():
    B = BTree(2)

    for i in range(10):
        B.insert((i, 2 * i))

    B.print_tree(B.root)

    if B.search_key(8) is not None:
        print("\nFound")
    else:
        print("\nNot Found")


    print('hello')

if __name__ == '__main__':
    main()
class TrieNode:
    """node in a trie structure"""
    def __init__(self, char):
        # identifier
        self._char = char

        # end of word (=leaf)
        self._is_leaf = True

        # dictionary of child nodes. keys = characters, values = nodes
        self._children = {}

    def get_char(self):
        return self._char

    def get_children(self):
        return self._children

    def get_child(self, char):
        """traverse down the trie. essentially return a child node corresponding to input char"""
        return self._children[char]

    def is_leaf(self):
        return self._is_leaf

    def add_child(self, node):
        char = node.get_char()
        self._children[char] = node
        self._is_leaf = False

    def set_leaf(self):
        self._is_leaf = True

class Trie:
    def __init__(self):
        """
        contain at least root node, does not store any characters
        """
        self._root = TrieNode('')

    def insert(self, word):
        """insert word into trie"""
        node = self._root

        # loop through each character in word, with every successful letter move down the trie
        for char in word:
            if char in node.get_children().keys():
                node = node.get_child(char)
            else:
                # character not found, create new node
                new_node = TrieNode(char)
                node.add_child(new_node)
                node = new_node

        # mark end of word
        node.set_leaf()

    def dfs(self, node, prefix):
        """
        search prefix using dfs
        node -      node to start with
        prefix -    current prefix, for tracing a word while traversing the trie
        """

        if node.is_leaf():
            self.output.append((prefix + node.get_char()))
            return self.output

        for child in node.get_children().values():
            self.dfs(child, prefix + node.get_char())

    def query(self, x):
        """
        given a prefix, retrieve all words stored in
        the trie with that prefix
        """

        self.output = []
        node = self._root

        # check if prefix in trie
        for char in x:
            if char in node.get_children().keys():
                node = node.get_child(char)
            else:
                # cannot find prefix, return empty list
                return []

        # got to here - prefix in trie. perform dfs from that node. TRIM LAST CHARACTER
        self.dfs(node, x[:-1])

        # sort results in reverse order
        return sorted(self.output, key=lambda x: x[1], reverse=True)

    def print(self):
        self.output = []
        self.dfs(self._root, '')

        print(self.output)

if __name__ == '__main__':
    t = Trie()
    t.insert("was")
    t.insert("word")
    t.insert("war")
    t.insert("what")
    t.insert("where")
    words = t.query("wh")
    t.print()

    print(words)
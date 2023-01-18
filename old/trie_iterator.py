from trie import TrieNode, Trie


class LFTrieNode(TrieNode):
    """trie node - where we also store list of keys in original relation.
    corresponds to an instance in some relation R(X,Y,Z,...) i.e. tuple t = (x,y,z,...)

    once we reach a leaf, we have an instance starting from root to that leaf. so, only leaves store TID and CNTs
    """
    def __init__(self, char, parent):
        # identifier
        self._char = char

        # end of word (=leaf)
        self._is_leaf = False

        # dictionary of child nodes. keys = characters, values = nodes
        self._children = {}

        # to support up(), we need to point to parent node
        self._parent = parent

    def get_parent(self):
        return self._parent

class LFTrie(Trie):
    def __init__(self):
        """
        contain at least root node, does not store any characters
        """
        self._root = LFTrieNode('', None)

    def insert(self, word):
        """insert tuple into trie"""
        node = self._root

        # loop through each character in the word
        for char in word:
            if char in node.get_children().keys():
                node = node.get_child(char)
            else:
                # character not found, create new node in trie
                new_node = LFTrieNode(char, node)
                node.add_child(new_node)
                node = new_node

        # mark end of word
        node.set_leaf()

    def get_root(self):
        return self._root


class LFTrieIterator():
    def __init__(self, trie):
        self._trie = trie
        self._current = trie.get_root()
        self._at_end = False
        self._at_end_depth = False   # reached end for current DEPTH

    def open(self):
        if self._at_end:
            print('-E- Cannot further open current branch. Try "next" instead')
            return

        first_child_key = list(self._current.get_children().keys())[0]
        self._current = self._current.get_child(first_child_key)

        self._at_end_depth = False

        if self._current.is_leaf():
            self._at_end = True

    def next(self):
        """
        proceed to next child of parent!
        """

        if self._at_end_depth:
            print('-E- Cannot further increment in current depth. Try "up,next,open" instead')
            return

        parent = self._current.get_parent()
        children = parent.get_children()


        # find next key in children dictionary
        res = None
        temp = iter(children)
        for char in temp:
            if char == self._current.get_char():
                res = next(temp, None)

        if res:
            self._current = parent.get_child(res)
        else:
            print('-E- Cannot further increment in current depth. Try "up,next,open" instead')
            self._at_end_depth = True

        # check if end of list
        try:
            next(temp)
        except StopIteration:
            self._at_end_depth = True


    def up(self):
        self._current = self._current.get_parent()
        self._at_end = False


    def get_key(self):
        """traverse to top of trie"""
        res = []
        node = self._current
        while node.get_parent():
            res.insert(0,node.get_char())
            node = node.get_parent()

        return ''.join(res)


if __name__ == '__main__':
    A = LFTrie()
    A.insert('1')
    A.insert('13')
    A.insert('134')
    A.insert('135')
    A.insert('146')
    A.insert('148')
    A.insert('149')
    A.insert('152')
    A.insert('352')
    it = LFTrieIterator(A)

    # get to 134
    it.open()
    it.open()
    it.open()
    print(it.get_key())

    # get to 135
    it.open()
    it.next()
    print(it.get_key())

    # get to 146
    it.next()
    it.up()
    it.next()
    it.open()
    print(it.get_key())
class iterator:
    def __init__(self, CNT, TID, attributes):
        self._attributes = list(attributes)
        self._CNT = CNT.sort_values(by=self._attributes)
        self._TID = TID.sort_values(by=self._attributes)

        self._iterator = self._TID[self._attributes].itertuples(index=False)
        self._key = next(self._iterator)
        self._at_end = False

    def key(self):
        """
        return key at current iterator position
        """
        return self._key

    def next(self):
        """
        proceed to next key
        """
        if self._at_end is True:
            print('-E- Iterator ended')
            return

        try:
            self._key = next(self._iterator)
        except StopIteration:
            self._at_end = True

    def seek(self, key):
        """
        position iterator at least upper bound to key
        i.e.
        either iterator is positioned at same key or larger key
        """
        while(self._key < key):
            self.next()

    def at_end(self):
        """
        true when iterator is at end
        """
        return self._at_end

    def get_indices(self):
        """
        based on current key position, get TID values (indices)
        """
        # fetch values for TID table
        value = self._key[0]
        attributes = self._attributes[0]
        # access TID table
        # works:
        # self._TID.loc[self._TID['B'] == '1']
        indices = self._TID.loc[self._TID[attributes] == value]['IDX'].to_list()
        return list(indices[0])
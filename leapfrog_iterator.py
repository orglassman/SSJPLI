from common import binary_search
from more_itertools.recipes import consume

class leapfrog_iterator():
    def __init__(self, table):
        """
        table structure:
        column 0 - possible values for attributes
        column 1 - list of instances

        instance index - runs over rows of table (instnaces)
        list index - for every instance, there's a list of indices in original relation. store index to that list
        """
        # external
        self._attributes = table.columns[0]

        # make everything hashable
        keys = table.iloc[:,0].to_list()
        values = table.iloc[:, 1].to_list()
        key_dict = dict(zip(keys, values))

        # remove empty lists
        clean_dict = {i:j for i,j in key_dict.items() if j != []}
        self._dict = clean_dict
        self._instance_iterator = iter(self._dict)

        # iterate over instances
        try:
            self._current_instance = next(self._instance_iterator)
            self._at_end_instance = False
        except StopIteration:
            self._at_end_instance = True
            print(f'-W- Empty iterator initialized for attributes {self._attributes}')
            return

        self.init_list()

        ################################################################
        # index = 0. key (in relation) = value itself
        ################################################################

    def init_list(self):
        # iterate over instance TIDs
        self._current_list = self._dict[self._current_instance]
        self._current_list_len = len(self._current_list)
        self._current_list_iterator = iter(self._current_list)
        self._at_end_list = False

        try:
            self._current_key = next(self._current_list_iterator)
        except StopIteration:
            self._at_end_list = True
            print(f'-W- Empty list for key {self._current_key} attributes {self._attributes}')
            return

        self._min_key = self._current_list[0]
        self._max_key = self._current_list[-1]

    ##### --------------------------------------------------------------
    #####   ATTRIBUTES -------------------------------------------------
    ##### --------------------------------------------------------------
    def __next__(self):
        """
        INCREMENTS LIST INDEX RATHER THAN INSTANCE INDEX
        """
        if self.is_end_list():
            print(f'-W- Reached end of list for instance {self.get_current_instance()} attributes {self.get_attributes()}')
            return

        try:
            self._current_key = next(self._current_list_iterator)
        except StopIteration:
            self._at_end_list = True
            print(f'-W- End of list for instance {self._current_instance} attributes {self._attributes}')

    def __lt__(self, other):
        self_key = self.get_key()
        other_key = other.get_key()
        return self_key < other_key

    def __eq__(self, other):
        self_key = self.get_key()
        other_key = other.get_key()
        return self_key == other_key

    def __gt__(self, other):
        self_key = self.get_key()
        other_key = other.get_key()
        return self_key > other_key

    def __le__(self, other):
        return (self < other) or (self == other)

    def __ge__(self, other):
        return (self > other) or (self == other)

    ##### --------------------------------------------------------------
    #####   GET --------------------------------------------------------
    ##### --------------------------------------------------------------
    def get_attributes(self):
        """
        return attributes as string
        i.e.
        [A,B] are treated as single "AB" rather than list [A, B]
        """
        return self._attributes

    def get_current_instance(self):
        """
        current instance which is handled by iterator
        """
        return self._current_instance

    def is_end_instance(self):
        """
        exhausted all instances in table
        """
        return self._at_end_instance

    def get_all_instances(self):
        return list(self._dict.keys())

    def get_current_list(self):
        return self._current_list

    def get_key(self):
        """
        get current key to which iterator is pointing
        """
        return self._current_key

    def is_end_list(self):
        """
        exhausted current instance (rows in original relation)
        """
        return self._at_end_list

    ##### --------------------------------------------------------------
    #####   MODIFIERS --------------------------------------------------
    ##### --------------------------------------------------------------
    def next_instance(self):
        """
        1. move on to the next instance of attributes
        2. restart list iterator
        3. set to at_end to False
        """
        if self._at_end_instance:
            print(f'-W- Already finished instances for attributes {self._attributes}')
            return

        try:
            self._current_instance = next(self._instance_iterator)
            self.init_list()
        except StopIteration:
            self._at_end_instance = True
            print(f'-W- Finished instances for attributes {self._attributes}')


    # TODO: B-Tree implementation
    def next_list(self):
        """
        increment list iterator
        """
        if self.is_end_list():
            print(f'-W- Already finished list for instance {self._current_instance} attributes {self._attributes}')
            return

        try:
            self._current_key = next(self._current_list_iterator)
        except StopIteration:
            self._at_end_list = True
            print(f'-W- Already reached end of list for instance {self.get_current_instance()} for attributes {self.get_attributes()}')

    def seek(self, key):
        """
        key - INTEGER

        seeking to increment list index ! not value
        """
        initial_index = binary_search(self._current_list, self._current_key)
        res = binary_search(self._current_list, key)

        # no need to increment iterator
        if initial_index ==  res:
            return

        # key not in list
        if res < 0:
            # key is too small - set index to 0
            if key < self._min_key:
                self._current_key = self._current_list[0]

            # key too big - cannot further seek current list
            elif key > self._max_key:
                self._at_end_list = True

            print(f'-W- Illegal seek key {key} where current key is {self._current_key}')
            return

        # res > 0 - found element. skip iterator to element
        consume(self._current_list_iterator, res - initial_index - 1)
        self._current_key = next(self._current_list_iterator)

    def set_list(self, instance):
        """
        fetch list for given instance
        """
        self._current_instance = instance
        self.init_list()
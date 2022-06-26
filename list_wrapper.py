class list_wrapper:
    """
    when storing lists of alternating sizes in a dataframe -
    store pointer to list instead list itself thus causing size issues for pandas

    wrap the list with this class and save an object instead original list
    """
    def __init__(self, list):
        self.list = list

    def get_list(self):
        return list

    def set_list(self, list):
        self.list = list
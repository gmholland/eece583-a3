class BucketList:
    """Class representing a Bucket List.
    
    Used to organize nodes by their gain value. Consists of an array of
    dictionaries. This array is implemented as a dictionary itself so that
    negative gain values may be used to indexes (keys)."""

    def __init__(self, pmax):
        self._bucketlist = {i : dict() for i in range(-pmax, pmax+1)}
        self._max_gain = -pmax
        self._pmax = pmax
        self._count = 0

    def is_empty(self):
        """Return True if the bucket list is empty, False otherwise."""
        if self._count == 0:
            return True
        else:
            return False

    def add_node(self, node):
        """Add a node with initial gain to the Bucket List."""
        # add node
        self._bucketlist[node.gain][node.ID] = node

        self._count += 1
        # update max gain pointer
        if node.gain > self._max_gain:
            self._max_gain = node.gain

    def _update_max_gain(self):
        """Update max gain if removing last node from the max gain bucket."""
        while not self._bucketlist[self._max_gain]:
            self._max_gain -= 1
            # make sure index does not go out of range
            if self._max_gain == -self._pmax:
                break

    def remove_node(self, node):
        """Remove a node from the Bucket List."""
        # remove node
        del self._bucketlist[node.gain][node.ID]
        self._update_max_gain()
        self._count -= 1

    def pop(self):
        """Return a node with highest gain and remove it."""
        (ID, node) = self._bucketlist[self._max_gain].popitem()

        self._update_max_gain()
        self._count -= 1
        return node

    def get_max_gain(self):
        return self._max_gain

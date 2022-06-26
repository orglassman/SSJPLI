import sys

class minheap:
    def __init__(self, size):
        self._storage = [0]*size
        self._size = size
        self._heap_size = 0
        self._Heap = [0] * (self._size + 1)
        self._Heap[0] = sys.maxsize * -1
        self._parent = 1
        self._root = 1

    def getParentIndex(self, index):
        return (index - 1) // 2

    def getLeftChildIndex(self, index):
        return 2 * index+1

    def getRightChildIndex(self, index):
        return 2 * index+2

    def hasParent(self, index):
        return self.getParentIndex(index) >= 0

    def insert(self, index):
        if self._heap_size >= self._size :
            return

        self._heap_size += 1
        self._Heap[self._heap_size] = index
        heap = self._heap_size
        while self._Heap[heap] < self._Heap[heap//2]:
            self.swap(heap, heap//2)
            heap = heap//2

    def swap(self, left, right):
        self._Heap[left], self._Heap[right] = self._Heap[right], self._Heap[left]

    def root_node(self, i):
        if not (i >= (self._heap_size//2) and i <= self._heap_size):
            if (self._Heap[i] > self._Heap[2 * i]  or  self._Heap[i] > self._Heap[(2 * i) + 1]):
                if self._Heap[2 * i] < self._Heap[(2 * i) + 1]:
                    self.swap(i, 2 * i)
                    self.root_node(2 * i)
                else:
                    self.swap(i, (2 * i) + 1)
                    self.min_heapify((2 * i) + 1)

    def getMin(self):
        min_value = self._Heap[self._root]
        self._Heap[self._root] = self._Heap[self._root]
        self._size-= 1
        self.root_node(self._root)
        return min_value

    def extractMin(self):
        data = self._Heap[1]
        self._Heap[1] = self._Heap[self._size-1]
        self._size -= 1
        return data
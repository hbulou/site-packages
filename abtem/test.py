class IndexingCache:

    def __init__(self, arr):
        self._arr = arr
        self._last_indexing_axis = None
        self._last_item = ()
        self._cached = None

    def __getitem__(self, item):
        indexing_axis = [last_i != i for last_i, i in zip(self._last_item, item)]
        print(indexing_axis)

        if sum(indexing_axis) == 1:
            indexing_axis = indexing_axis.index(True)
        else:
            indexing_axis = 0

        if not self._last_item:
            print("compute")
            self._last_item = item
            return self._arr[item]

        if indexing_axis == self._last_indexing_axis:
            print("use cached")
            self._last_item = item
            return self._cached[item[indexing_axis]]
        else:
            print("compute and cache")
            self._last_indexing_axis = indexing_axis
            self._last_item = item

            cache_item = tuple(slice(None) if i == indexing_axis else j for i, j in enumerate(item))

            self._cached = self._arr[cache_item]

            return self._cached[item[indexing_axis]]
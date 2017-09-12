import numpy as np


class DataSet:
    def __init__(self, x: np.ndarray, y: np.ndarray, shuffle=True):

        self.n_samples = x.shape[0]
        if shuffle:
            _idx = np.random.permutation(self.n_samples)
            self.x = x[_idx, :]
            self.y = y[_idx, :]
        else:
            self.x = x
            self.y = y

        self.dim_x = np.prod(x.shape[1:])
        self.dim_y = np.prod(y.shape[1:])

    def get_y_1hot(self):
        n_class = self.get_y_categories().ravel().shape[0]
        y_ = np.eye(n_class)[self.y.astype(np.int8).ravel()]
        return y_

    def get_y_categories(self):
        return np.unique(self.y.ravel())

    def get_x_by_y(self, y):
        idx = np.ravel(self.y == y)
        return self.x[idx, :]

    def subset(self, index, shuffle=False):
        return DataSet(self.x[index, :], self.y[index, :], shuffle)

    def sample(self, size):
        _idx = np.random.randint(self.n_samples, size=size)
        return self.subset(_idx, shuffle=True)

    def random_split(self, ratio):
        part_1_size = int(ratio * self.n_samples)
        part_1_idx = np.random.choice(self.n_samples, size=part_1_size, replace=False)
        part_2_idx = [i for i in range(self.n_samples) if i not in part_1_idx]
        return self.subset(part_1_idx), self.subset(part_2_idx)

    def __add__(self, d2, shuffle=True):
        if not (self.dim_x == d2.dim_x
                and self.dim_y == d2.dim_y):
            raise ValueError("Dimension mismatch")
        else:
            x_new = np.vstack((self.x, d2.x))
            y_new = np.vstack((self.y, d2.y))
            return DataSet(x_new, y_new, shuffle=shuffle)

    def __radd__(self, d2, shuffle=True):
        if d2 == 0:
            return self
        else:
            return self.__add__(d2, shuffle=shuffle)

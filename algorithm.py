import numpy as np

class KMeans:
    def __init__(self, data, k, init_method='random'):
        self.data = data
        self.k = k
        self.assignment = [-1 for _ in range(len(data))]
        self.centers = None
        self.iteration_complete = False
        self.init_method = init_method

    def initialize_centers(self):
        if self.init_method == 'random':
            return self.random_init()
        elif self.init_method == 'farthest_first':
            return self.farthest_first_init()
        elif self.init_method == 'kmeans++':
            return self.kmeans_plus_plus_init()
        elif self.init_method == 'manual':
            # For manual initialization, you'll need to implement a mechanism to get user input
            # For now, we'll use random initialization as a placeholder
            return self.random_init()
        else:
            raise ValueError("Invalid initialization method")

    def random_init(self):
        return self.data[np.random.choice(len(self.data), size=self.k, replace=False)]

    def farthest_first_init(self):
        centers = [self.data[np.random.choice(len(self.data))]]
        while len(centers) < self.k:
            distances = np.array([min([self.dist(c, x) for c in centers]) for x in self.data])
            centers.append(self.data[np.argmax(distances)])
        return np.array(centers)

    def kmeans_plus_plus_init(self):
        centers = [self.data[np.random.choice(len(self.data))]]
        while len(centers) < self.k:
            distances = np.array([min([self.dist(c, x) for c in centers]) for x in self.data])
            probs = distances ** 2 / np.sum(distances ** 2)
            cumprobs = np.cumsum(probs)
            r = np.random.random()
            i = np.searchsorted(cumprobs, r)
            centers.append(self.data[i])
        return np.array(centers)

    def make_clusters(self, centers):
        # Assign points to the nearest centroid
        for i in range(len(self.data)):
            distances = [self.dist(centers[j], self.data[i]) for j in range(self.k)]
            self.assignment[i] = int(np.argmin(distances))

    def compute_centers(self):
        new_centers = np.zeros((self.k, len(self.data[0])))
        for i in range(self.k):
            points_in_cluster = self.data[np.array(self.assignment) == i]
            if len(points_in_cluster) > 0:
                new_centers[i] = np.mean(points_in_cluster, axis=0)
            else:
                # If a cluster is empty, reinitialize its center randomly
                new_centers[i] = self.data[np.random.choice(len(self.data))]
        return new_centers

    def dist(self, x, y):
        return np.linalg.norm(x - y)

    def step(self):
        if self.iteration_complete:
            return None
        if self.centers is None:
            self.centers = self.initialize_centers()
            self.make_clusters(self.centers)
            return self.centers, self.assignment
        # Perform one iteration of Lloyd's algorithm
        new_centers = self.compute_centers()
        self.make_clusters(new_centers)  # Use new_centers here instead of self.centers
        if np.array_equal(self.centers, new_centers):
            self.iteration_complete = True
        else:
            self.centers = new_centers
        return self.centers, self.assignment
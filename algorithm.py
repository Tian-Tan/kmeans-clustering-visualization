import numpy as np

class KMeans:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.assignment = [-1 for _ in range(len(data))]
        self.centers = None
        self.iteration_complete = False

    def initialize_centers(self):
        # Randomly initialize centroids from the data points
        return self.data[np.random.choice(len(self.data), size=self.k, replace=False)]

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

        # Perform one iteration of Lloyd's algorithm
        new_centers = self.compute_centers()
        self.make_clusters(new_centers)  # Use new_centers here instead of self.centers
        if np.array_equal(self.centers, new_centers):
            self.iteration_complete = True
        else:
            self.centers = new_centers
        return self.centers, self.assignment
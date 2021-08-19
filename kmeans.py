import numpy as np
import matplotlib.pyplot as plt

# select random points from data as initial centroids
def initialize_centroids(points, k, pop_size):
    # choosing randomly k cluster centroids 
    centroids = points[np.random.choice(pop_size, k, replace=False)]
    centroids = {i+1:centroids[i] for i in range(k)}
    
    return centroids

# calculate distance between two points according to euclidean distance    
def calculate_distance(x, y):
    return np.sqrt(sum((x-y)**2))
    
# assign points to nearest cluster    
def assign_points_to_clusters(points, centroids, clusters):
    
    for order, point in enumerate(points):
        
        min_dist = 9999999.9
        min_distanced_center = clusters[order]
        
        for key, center in centroids.items():
        
            dist = calculate_distance(point, center)
           
            if dist <= min_dist:
                min_distanced_center = key
                min_dist = dist
        
        clusters[order] = int(min_distanced_center)
    
    return clusters
    
    
# calculate new centroids  
def calculate_centroids(points, centroids, clusters):
          
    for key in centroids.keys():
        
        new_mean = np.mean(points[clusters==key], axis=0)
        centroids[key] = new_mean
        
    return centroids

# algorithm steps    
def kmeans(points, k):
    
    pop_size = len(points)
    
    clusters = np.zeros(pop_size, dtype=int)
    centroids = initialize_centroids(points, k, pop_size)
    
    prev_error = 0
    iter_num = 1
    while True:

        clusters = assign_points_to_clusters(points, centroids, clusters)
        centroids = calculate_centroids(points, centroids, clusters)

        total_error = sum([calculate_distance(centroids[cluster], point) for cluster, point in zip(clusters, points)])

        if total_error == prev_error: break

        prev_error = total_error
        iter_num +=1
    
    return clusters, centroids
    
    

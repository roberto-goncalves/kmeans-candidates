import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from pandas.core.common import flatten
import pandas as pd
import numpy as np
import collections
import random
import itertools
import math

class InitKMeans(object):
  """Apply KMeans to a dataframe.
  
  """
  def __init__(self, cluster_number):
    """ Constructor

    Args:
      cluster_number: Number of clusters to compute.
    """
    self._kmeans_df = None
    self._cluster_number = cluster_number
    self._kmeans_result = None
    self._models_df = None
    self._sequential_index = None
    self._cluster_centers = None

  def computeKMeans(self, source_df):
    """Apply transformations to the source_df and add columns to be processed

    Args:
      source_df: Dataframe to compute KMeans on.
    """
    self._kmeans_df = source_df.copy()
    self._kmeans_result = self.kmeans(self._kmeans_df.select_dtypes(include='float'), cluster_number=self._cluster_number)
    self._cluster_centers = self._kmeans_result.cluster_centers_
    self.insertAdjcentColumns(source_df=source_df)

  def insertAdjcentColumns(self, source_df):
    """Insert/compute kmeans columns and essential columns such as, points and distances.

    Args:
      source_df: Dataframe to compute KMeans on.
    """
    self.insertClusterLabels(df=source_df)
    self.insertClusterCenters(df=source_df, cluster_centroids_map=self.createClusterCentroidsMap(self._kmeans_result))
    self.insertPoints(df=source_df)
    self.insertDistances(df=source_df)

  def computePoints(self, source_df, number_of_points, distance="closest"):
    """ Compute the number of points.

    Args:
      source_df: Dataframe to compute KMeans on.
      number_of_points: Number of points to compute.
      distance: Distance to compute.
    """
    if number_of_points > len(source_df):
      raise ValueError('The number of points must be less than the number of rows')
    if number_of_points <= 0:
      raise ValueError('The number of points must be greater than 0')
    if distance != "closest" and distance != "farthest":
      raise ValueError('The distance must be closest or farthest')

    if distance == "closest":
      return self.computeClosestPoints(source_df, number_of_points)
    elif distance == "farthest":
      return self.computeFarthestPoints(source_df, number_of_points)

  def computeClosestPoints(self, source_df, number_of_points):
    """ Compute the closest points

    Args:
      source_df: Dataframe to compute KMeans on.
      number_of_points: Number of points to compute.
    """
    sorted_df = self.sortDataframe(source_df, 'distance_to_centroid', ascending=True)
    label_distance_map = self.createLabelDistanceMap(df=sorted_df)
    return self.getPoints(number_of_points, label_distance_map)

  def computeFarthestPoints(self, source_df, number_of_points):
    """ Compute the farthest points

    Args:
      source_df: Dataframe to compute KMeans on.
      number_of_points: Number of points to compute.
    """
    sorted_df = self.sortDataframe(source_df, 'distance_to_centroid', ascending=False)
    label_distance_map = self.createLabelDistanceMap(df=sorted_df)
    return self.getPoints(number_of_points, label_distance_map)

  def sortDataframe(self, source_df, desired_column, ascending):
    """ Sort the dataframe

    Args:
      source_df: Dataframe to compute KMeans on.
      desired_column: Column to sort.
      ascending: Sort ascending or descending.
    """
    return source_df.sort_values(by=desired_column, ascending=ascending)

  def getPoints(self, number_of_points, label_distance_map):
    """ Retrieve points by processing cluster_label within label_distance_map.
    This process goes through each cluster_label, removing the first element from the deque.
    Args:
      number_of_points: Number of points to compute.
      label_distance_map: Label distance map.
    Returns:
      List of points.
    """
    selected_points = []
    while number_of_points > 0 and any(label_distance_map.values()):  # Check if any deques have elements left
      for cluster_label in range(self._cluster_number):
        if label_distance_map[cluster_label]:
          selected_points.append(label_distance_map[cluster_label].popleft())
          number_of_points -= 1
          if number_of_points == 0:
            break  # Exit early if enough points are selected
    return selected_points

  def createLabelDistanceMap(self, df):
    """ Create label distance map
    Args:
      df: Dataframe to compute KMeans on.
    Returns:
      Label distance map.
    """
    label_distance_map = collections.defaultdict(collections.deque)
    for index, row in df.iterrows():
      label = int(row["cluster_label"])
      # Assuming number_of_clusters of 3. {cluster_label_0: [[habitat1 attributes], [habitat2 attributes]], cluster_label_1: [habitat3 attributes] ...}
      habitat_data = [row["habitat"], row["x"], row["y"], row["z"], row["distance_to_centroid"], label]
      label_distance_map[label].append(habitat_data)
    return label_distance_map

  def insertPoints(self, df):
    """ Add columns as a single column called points in df.
    Args:
      df: Dataframe to compute KMeans on.
    Returns:
      Label distance map.
    """
    float_df = df.copy().select_dtypes(include='float')
    df.insert(len(df.columns), 'point', [list(row) for index, row in float_df.iterrows()])

  def calcDistance(self, p1, p2):
    """Calculate the distance between two points

    Args:
      p1: Point 1
      p2: Point 2
    Returns:
      distance between p1 and p2
    """
    return math.sqrt(sum((p1-p2)**2))

  def insertDistances(self, df):
    """Compute and Insert distances to centroid to df

    Args:
      df: Dataframe to compute KMeans on.
    """
    points = np.array(df['point'].tolist())
    cluster_centers = np.array(df['cluster_centers'].tolist())
    distances = []
    for index in range(points.shape[0]):
      distances.append(self.calcDistance(points[index,], cluster_centers[index, ]))
    df.insert(len(df.columns), 'distance_to_centroid', distances)

  def kmeans(self, df, cluster_number):
    """Call sklearn KMeans

    Args:
      df: Dataframe to compute KMeans on.
      cluster_number: Number of clusters to compute.
    Returns:
      KMeans result.
    """
    return KMeans(n_clusters=cluster_number, init='k-means++', max_iter=400, n_init=10, random_state=0).fit(df)

  def insertClusterLabels(self, df):
    """Insert cluster labels into dataframe.

    Args:
      df: Dataframe to insert cluster labels into.
    """
    df.insert(len(df.columns), "cluster_label", self._kmeans_result.labels_)

  def insertClusterCenters(self, df, cluster_centroids_map):
    """ Insert cluster centroids into dataframe.

    Args:
      df: Dataframe to insert cluster centroids into.
      cluster_centroids_map: Cluster centroids map.
    """
    df.insert(len(df.columns), "cluster_centers", [cluster_centroids_map[cluster] for cluster in df["cluster_label"]])

  def createClusterCentroidsMap(self, kmeans_result):
    """ Create cluster centroids map
    The map has the following format: {0: [centroid coordinates], 1: [centroid coordinates] ...}
    Args:
      kmeans_result: KMeans result.
    Returns:
      Cluster centroids map.
    """
    return {cluster: centroids for cluster, centroids in zip(set(sorted(kmeans_result.labels_)), kmeans_result.cluster_centers_)}

  def plot(self, df, closest_habitats, farthest_habitats):
    """ Plot habitat points.

    Args:
      df: Dataframe to plot.
      closest_persons: Closest habitat points.
      farthest_persons: Farthest habitat points.
    """
    fig = plt.figure(figsize = (12, 12))
    ax = plt.axes(projection='3d')
    
    closest_x, closest_y, closest_z = zip(*([closest[1:4] for closest in closest_habitats])) 
    ax.scatter(closest_x, closest_y, closest_z, c="purple", alpha=1.0, label="closest")
    farthest_x, farthest_y, farthest_z = zip(*([farthest[1:4] for farthest in farthest_habitats]))
    ax.scatter(farthest_x, farthest_y, farthest_z, c="black", alpha=1.0, label="farthest")
    color_arr = ["red", "green", "blue"]
    for i in range(self._cluster_number):
        cluster_data = df[df['cluster_label'] == i]
        # Plot habitats
        ax.scatter(cluster_data['x'], cluster_data['y'], cluster_data['z'], 
                   c=color_arr[i], alpha=0.2, label="habitat" if i == 0 else "")  # Label only once
        # Plot centroid
        cluster_center = cluster_data['cluster_centers'].iloc[0]
        ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], 
                   c=color_arr[i], alpha=1.0, label="centroid-habitat" if i == 0 else "") 
    ax.set_xlabel('Average Temperature')
    ax.set_ylabel('Number of Animals')
    ax.set_zlabel('Moisture')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # Creating habitats dataset
    data = np.random.randint(0, 41, size=(130, 4))
    test_df = pd.DataFrame(data, columns=['habitat', 'x', 'y', 'z'])
    test_df[['x', 'y', 'z']] = test_df[['x', 'y', 'z']].astype(float)
    kmeansClusters = InitKMeans(cluster_number=3)
    # Computing KMeans
    kmeansClusters.computeKMeans(test_df)
    # Computing closest and farthest
    closest_habitats = kmeansClusters.computePoints(source_df=test_df, number_of_points=3, distance='closest')
    farthest_habitats = kmeansClusters.computePoints(source_df=test_df, number_of_points=3, distance='farthest')
    # Excluding closest and farthest from the current dataset
    test_df = test_df[~test_df['habitat'].isin([row[0] for row in closest_habitats])]
    test_df = test_df[~test_df['habitat'].isin([row[0] for row in farthest_habitats])]
    # Plot dataframes
    kmeansClusters.plot(test_df, closest_habitats, farthest_habitats)
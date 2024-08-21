import unittest
import pandas as pd
from kmeans import InitKMeans

class TestKMeans(unittest.TestCase):
  def setUp(self):
    self.test_df = df = pd.DataFrame({"habitat": range(15), "x": range(15), "y": range(15), "z": range(15)})
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["z"] = df["z"].astype(float)

  def testClosestK3MeansAll(self):
    kmeansClusters = InitKMeans(cluster_number=3)
    kmeansClusters.computeKMeans(self.test_df)
    closest_habitats = kmeansClusters.computePoints(source_df=self.test_df, number_of_points=15, distance='closest')
    assert closest_habitats == [[12, 12.0, 12.0, 12.0, 0.0, 0], [2, 2.0, 2.0, 2.0, 0.8660254037844386, 1], [7, 7.0, 7.0, 7.0, 0.8660254037844386, 2], [11, 11.0, 11.0, 11.0, 1.7320508075688772, 0], [3, 3.0, 3.0, 3.0, 0.8660254037844386, 1], [8, 8.0, 8.0, 8.0, 0.8660254037844386, 2], [13, 13.0, 13.0, 13.0, 1.7320508075688772, 0], [1, 1.0, 1.0, 1.0, 2.598076211353316, 1], [6, 6.0, 6.0, 6.0, 2.598076211353316, 2], [10, 10.0, 10.0, 10.0, 3.4641016151377544, 0], [4, 4.0, 4.0, 4.0, 2.598076211353316, 1], [9, 9.0, 9.0, 9.0, 2.598076211353316, 2], [14, 14.0, 14.0, 14.0, 3.4641016151377544, 0], [0, 0.0, 0.0, 0.0, 4.330127018922194, 1], [5, 5.0, 5.0, 5.0, 4.330127018922194, 1]]

  def testFarthestK3MeansAll(self):
    kmeansClusters = InitKMeans(cluster_number=3)
    kmeansClusters.computeKMeans(self.test_df)
    farthest_habitats = kmeansClusters.computePoints(source_df=self.test_df, number_of_points=15, distance='farthest')
    assert farthest_habitats == [[10, 10.0, 10.0, 10.0, 3.4641016151377544, 0], [0, 0.0, 0.0, 0.0, 4.330127018922194, 1], [6, 6.0, 6.0, 6.0, 2.598076211353316, 2], [14, 14.0, 14.0, 14.0, 3.4641016151377544, 0], [5, 5.0, 5.0, 5.0, 4.330127018922194, 1], [9, 9.0, 9.0, 9.0, 2.598076211353316, 2], [11, 11.0, 11.0, 11.0, 1.7320508075688772, 0], [1, 1.0, 1.0, 1.0, 2.598076211353316, 1], [7, 7.0, 7.0, 7.0, 0.8660254037844386, 2], [13, 13.0, 13.0, 13.0, 1.7320508075688772, 0], [4, 4.0, 4.0, 4.0, 2.598076211353316, 1], [8, 8.0, 8.0, 8.0, 0.8660254037844386, 2], [12, 12.0, 12.0, 12.0, 0.0, 0], [2, 2.0, 2.0, 2.0, 0.8660254037844386, 1], [3, 3.0, 3.0, 3.0, 0.8660254037844386, 1]]

  def testFailureGreater(self):
    kmeansClusters = InitKMeans(cluster_number=6)
    kmeansClusters.computeKMeans(self.test_df)
    with self.assertRaises(ValueError):
      kmeansClusters.computePoints(source_df=self.test_df, number_of_points=40, distance='closest')

  def testFailureZero(self):
    kmeansClusters = InitKMeans(cluster_number=6)
    kmeansClusters.computeKMeans(self.test_df)
    with self.assertRaises(ValueError):
      kmeansClusters.computePoints(source_df=self.test_df, number_of_points=0, distance='closest')

unittest.main(argv=['first-arg-is-ignored'], exit=False)
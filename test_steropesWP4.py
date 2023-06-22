import os
import csv
import unittest
import numpy as np
import tempfile

from steropesWP4 import calculate_cluster_percentages, change_colorspace


class MyTestCase(unittest.TestCase):

    def test_calculate_cluster_percentages(self):
        # Create dummy cluster images
        cluster1_img = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])

        cluster2_img = np.array([[1, 0, 1],
                                 [0, 0, 0],
                                 [1, 0, 1]])

        # Call the function to calculate cluster percentages
        percentages = calculate_cluster_percentages(cluster1_img, cluster2_img)

        # Check the expected percentages
        expected_percentages = {
            'Cluster 1': 27.77777777777778,
            'Cluster 2': 22.22222222222222
        }

        # Assert that the calculated percentages match the expected percentages
        self.assertEqual(percentages, expected_percentages)



    def test_change_colorspace(self):

        # Create dummy image
        img = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])

        # Call the function to change the colorspace
        # Create a temporary output folder
        with tempfile.TemporaryDirectory() as temp_folder:
            # Call the function to change the colorspace

            try:
                img = change_colorspace(img, False, temp_folder, "output_img", "HSV")
            except Exception as e:
                print(f"Error occurred during colorspace conversion: {e}")
    
        # Check the expected image
        expected_img = np.array([[0, 255, 0],
                                 [255, 255, 255],
                                 [0, 255, 0]])

        # Assert that the calculated percentages match the expected percentages
        self.assertEqual(img.all(), expected_img.all())



if __name__ == '__main__':
    unittest.main(verbose=2)

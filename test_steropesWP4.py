import unittest
import os
import cv2
import numpy as np

from steropesWP4 import (
    imread_func,
    im2double_func,
    change_colorspace,
    kmeans_clustering,
    fuzzy_cmans_automatic_Th,
    calculate_cluster_percentages,
    calculate_cluster_pixel_counts,
)


class TestYourScript(unittest.TestCase):
    def setUp(self):
        # Set up any necessary resources or configurations before each test
        self.test_image_path = 'exampleFiles/K_1_62_2.JPG'

    def test_imread_func(self):
        # Test imread_func function
        img = imread_func(self.test_image_path, True)
        self.assertIsInstance(img, np.ndarray)

    def test_im2double_func(self):
        # Test im2double_func function
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        double_img = im2double_func(img)
        self.assertTrue(np.max(double_img) <= 1.0)

    def test_change_colorspace(self):
        # Test change_colorspace function
        img = np.random.rand(100, 100, 3)
        converted_img = change_colorspace(img, False, '', '', 'hsv')
        self.assertIsInstance(converted_img, np.ndarray)

    def test_kmeans_clustering(self):
        # Test kmeans_clustering function
        img = np.random.rand(100, 100, 3)
        clustered_img = kmeans_clustering(img, 2, 'random', False, 'test', False)
        self.assertIsInstance(clustered_img, np.ndarray)

    def test_fuzzy_cmans_automatic_Th(self):
        # Replace this path with the actual path to your test image
        img_path = 'exampleFiles/K_1_62_2.JPG'
        rotate_to_original = True  # Adjust this based on your requirement

        img = imread_func(img_path, rotate_to_original)

        # Call the fuzzy_cmans_automatic_Th function
        result = fuzzy_cmans_automatic_Th(img, 2, False, 'test')

        if result is not None:
            cluster1_img, cluster2_img, cntr, u = result

            # Add assertions for shapes
            self.assertEqual(cluster1_img.shape, (expected_shape1))
            self.assertEqual(cluster2_img.shape, (expected_shape2))
            # Add more assertions as needed...

            # Rest of your test code for fuzzy_cmans_automatic_Th...
        else:
            print("Error: fuzzy_cmans_automatic_Th returned None. Check logs for details.")
            self.fail("fuzzy_cmans_automatic_Th failed")



    def test_calculate_cluster_percentages(self):
        # Test calculate_cluster_percentages function
        u = np.random.rand(2, 100)
        total_pixels = 100
        percentages = calculate_cluster_percentages(u, total_pixels)
        self.assertIsInstance(percentages, dict)

    def test_calculate_cluster_pixel_counts(self):
        # Test calculate_cluster_pixel_counts function
        u = np.random.rand(2, 100)
        counts = calculate_cluster_pixel_counts(u)
        self.assertIsInstance(counts, dict)

if __name__ == '__main__':
    unittest.main()

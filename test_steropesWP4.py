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
        # Test for fuzzy_cmans_automatic_Th function
        img = imread_func(self.test_image_path, True)
        img_double = im2double_func(img)

        # Call the fuzzy_cmans_automatic_Th function
        cluster_images, cntr = fuzzy_cmans_automatic_Th(img_double, 2, False, 'test')

        self.assertIsNotNone(cluster_images)
        self.assertTrue(len(cluster_images) > 0)
        for cluster_img in cluster_images:
            self.assertIsInstance(cluster_img, np.ndarray)
            self.assertEqual(cluster_img.shape, img_double.shape[:-1])

        self.assertIsInstance(cntr, np.ndarray)


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

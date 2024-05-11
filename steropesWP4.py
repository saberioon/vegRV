"""
SteropesWP4.py\n
Author: Mohammadmhedi Saberioon
email: mohammadmehdi.saberioon@ilvo.vlanderen.be \n
Description: This code is developed for the WP4 of the Steropes project.
The main goal of this code is to segment the vegetation from the RGB images. \n 
Date: 2023-06-22  \n
update: 2023-11-16
"""




import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from skimage import filters, exposure
import argparse
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os
#from sklearn.cluster import DBSCAN
from sklearn.utils import shuffle
from sklearn.mixture import GaussianMixture


# -------------- These two libraris are for showing the progress------------------
import sys
import time
import os
#import json
import csv
# --------------for logging all messages ---------------
import datetime

from version import version

print(f"vegRV v{version}")

dt = datetime.datetime.now()
tmark = dt.strftime("%Y-%m-%d %H:%M:%S")

LOG_FILENAME = "steropes_WP4"+tmark+".log"

# ---------------------------------------------

def parse_arg():
    parser = argparse.ArgumentParser(prog='steropesWP4.py',description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", dest='input', type=str, help="input folder", required=True)
    parser.add_argument("-o", "--output", dest='output', type=str, help="output folder ", required=True)
    parser.add_argument("-c", "--colorspace", dest='colorspace', type=str, help="colorspace hsv, hls, yiq ", default='hls') 

    return parser


# ------------------------------------------------logger function-------------------------

def log_message(msg):
    dt = datetime.datetime.now()
    tmark = dt.strftime("%Y-%m-%d %H:%M:%S")

    logmsg = "{}: {}".format(tmark, msg)
    #log_filename = os.path.join(output_folder, "log.txt")
    # Log message to file
    with open(LOG_FILENAME, "a") as fw:
        fw.write(logmsg)
        fw.write('\n')

    # Show message in terminal
    print(logmsg)



# ------------------------------------------------Progress bar function-------------------------
def update_progress(prefix, progress):
    ''' progress in range (0, 1) '''
    points = 50
    done = int(progress * points)
    rest = points - done
    sys.stdout.write("\r{} [{}{}] {:4.1f}%".format(prefix, "#" * done, ' ' * rest, progress * 100))
    sys.stdout.flush()

# ------------------------------------------------


def imread_func(pathfile, Rotate_to_Original): 
  '''
    Rotate_to_Original must be either True or False

    cv2.imread uses BGR format to read data
    plt.imread uses RGB format to read data. However, plt flips the image. Thus, we use [::-1,::-1] to rotate image to the original format.

  '''
  if Rotate_to_Original:
    img = plt.imread(pathfile)[::-1,::-1]
  else:
    img = plt.imread(pathfile)
  return img


def im2double_func(img):
  '''
      output = (inputimage-min)/(max-min)
  '''
  out = img/255

  return out


def change_colorspace(img, to_save_or_not, output_folder, output_save_name, colorspace):
    """
    Convert the color space of the input image.

    Args:
        img (numpy.ndarray): Input image array.
        to_save_or_not (bool): Flag indicating whether to save the converted image.
        output_folder (str): Output folder path for saving the converted image.
        output_save_name (str): Output save name for the converted image.
        colorspace (str): Color space to convert the image to. Supported values: 'hsv', 'hls', 'yiq'.

    Returns:
        numpy.ndarray: The converted image array.

    Raises:
        ValueError: If the specified colorspace is not supported.
    """
    log_message("Colorspace conversion to {}...".format(colorspace))
    converted_img = np.zeros_like(img)

    for i in tqdm(range(img.shape[0]), desc="Colorspace Conversion Progress", ncols=80, mininterval=1.0):
        for j in range(img.shape[1]):
            r, g, b = img[i, j]

            if colorspace == 'hsv':
                converted_img[i, j] = colorsys.rgb_to_hsv(r, g, b)
            elif colorspace == 'hls':
                converted_img[i, j] = colorsys.rgb_to_hls(r, g, b)
            elif colorspace == 'yiq':
                converted_img[i, j] = colorsys.rgb_to_yiq(r, g, b)

    if to_save_or_not:
        filename = os.path.join(output_folder, output_save_name + '_' + colorspace + '.jpg')
        cv2.imwrite(filename, converted_img[::-1, ::-1] * 255)

    log_message("Colorspace conversion to {} completed".format(colorspace))
    return converted_img




def find_optimal_clusters_silhouette(data, max_clusters=10):
    """
    Find the optimal number of clusters based on silhouette score.

    Args:
        data (numpy.ndarray): Input data for clustering.
        max_clusters (int): Maximum number of clusters to consider.

    Returns:
        int: Optimal number of clusters based on silhouette score.
    """

    log_message("Finding the optimal number of clusters ...")

    silhouette_scores = []

    # Loop over different numbers of clusters
    with tqdm(total=max_clusters - 1, desc="Finding Clusters", ncols=80, mininterval=1.0) as pbar:
        for i in range(2, max_clusters + 1):
            # Fit KMeans with 'i' clusters
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            labels = kmeans.fit_predict(data)

            # Calculate silhouette score
            silhouette = silhouette_score(data, labels)
            silhouette_scores.append(silhouette)

            pbar.update(1)

    # Choose the number of clusters based on the highest silhouette score
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    log_message("Finding the optimal number of clusters completed")
    log_message("The optimal number of clusters are {}".format(optimal_clusters))

    return optimal_clusters




def kmeans_clustering(img, n_of_clusters, init_value='k-means++', to_save_or_not=True, output_save_name=None, verbose=True, n_init=10, n_init_value ='auto'):
    """
    Perform k-means clustering on the input image.

    Args:
        img (numpy.ndarray): Input image array.
        n_of_clusters (int): Number of clusters to create.
        init_value (str): Initialization method for cluster centers. Supported values: 'random', 'k-means++'.
        to_save_or_not (bool): Flag indicating whether to save the clustered image.
        output_save_name (str): Output save name for the clustered image.
        verbose (bool): Whether to display progress bars.
        n_init (int): Number of times the k-means algorithm will be run with different centroid seeds.

    Returns:
        tuple: A tuple containing the clustered image array, cluster assignments, and cluster centers (if verbose=True).
    """
    if verbose:
        log_message("kmeans clustering ...")

    kmeans = KMeans(n_clusters=n_of_clusters, init=init_value, n_init= n_init_value)

    # Flatten the image for clustering
    flattened_img = np.reshape(img, (-1, img.shape[2]))

    # Create a progress bar for iterations
    with tqdm(total=kmeans.max_iter * n_of_clusters, desc="KMeans Clustering Progress", ncols=80, mininterval=1.0, disable=not verbose) as pbar:
        kmeans.fit(flattened_img)

        # Create a progress bar for saving the clustered image
        with tqdm(total=img.shape[0], desc="Saving Clustered Image", ncols=80, mininterval=1.0, disable=not verbose) as save_pbar:
            clustered_img = np.reshape(kmeans.predict(flattened_img), img.shape[:-1])

            if to_save_or_not:
                filename = output_save_name + '_clustered.jpg'
                cv2.imwrite(filename, clustered_img[::-1, ::-1])

                save_pbar.update(1)

            pbar.update(kmeans.n_iter_ * n_of_clusters)

    if verbose:
        log_message("kmeans clustering completed")

    if verbose:
        return clustered_img, kmeans.labels_, kmeans.cluster_centers_
    else:
        return clustered_img




def fuzzy_cmans_automatic_Th(img, n_of_clusters, to_save_or_not, output_save_name):
    """
    Perform fuzzy c-means clustering on the input image.

    Args:
        img (numpy.ndarray): Input image array.
        n_of_clusters (int): Number of clusters for the clustering algorithm.
        to_save_or_not (bool): Flag indicating whether to save the resulting cluster images.
        output_save_name (str): Output save name for the cluster images.

    Returns:
        tuple: A tuple containing the cluster images and the cluster centers.
    """

    log_message("fuzzy cmeans clustering ...")
    img_shape = img.shape
    if len(img_shape) < 3:
        img = np.expand_dims(img, axis=2)

    flattened_img = np.reshape(img, (-1, img.shape[2]))

    # Perform fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        flattened_img.T, n_of_clusters, 2, error=0.005, maxiter=1000, init=None
    )

    # Ensure that the shape of u is correct
    if u.shape[1] != flattened_img.shape[0]:
        raise ValueError("Unexpected shape of the u array from fuzzy c-means.")

    # Reshape u to match the original image shape for each cluster
    u_reshaped = u.reshape(n_of_clusters, img.shape[0], img.shape[1])

    cluster_images = []
    for i in range(n_of_clusters):
        cluster_img = np.zeros((img.shape[0], img.shape[1]))
        threshold = filters.threshold_otsu(u_reshaped[i])

        # Apply threshold to create a binary image for each cluster
        cluster_img[u_reshaped[i] >= threshold] = 1

        if to_save_or_not:
            filename = f"{output_save_name}_cluster{i + 1}.jpg"
            cv2.imwrite(filename, (cluster_img * 255).astype(np.uint8))

        cluster_images.append(cluster_img)

    return cluster_images, cntr


def perform_gmm_segmentation(converted_img, min_clusters=2, max_clusters=5):
    """
    Perform Gaussian Mixture Model (GMM) segmentation on the input image.

    Args:
        converted_img (numpy.ndarray): Input image array in the desired color space.
        min_clusters (int): Minimum number of clusters for GMM.
        max_clusters (int): Maximum number of clusters for GMM.

    Returns:
        tuple: A tuple containing the best GMM model, the segmented image, and the best number of clusters.
    """
    log_message("GMM segmentation ...")

    # Reshape image dimension for processing
    image_reshaped = np.reshape(converted_img, (-1, 3))
    image_reshaped = shuffle(image_reshaped)

    # Initialize variables to store best GMM and its silhouette score
    best_gmm = None
    best_silhouette_score = -1
    best_n_clusters = None

    # Create a progress bar for the GMM fitting process
    with tqdm(total=max_clusters - min_clusters + 1, desc="GMM Fitting Progress", ncols=80, mininterval=1.0) as pbar:
        # Fit Gaussian Mixture Model (GMM) to each number of clusters
        for n_clusters in range(min_clusters, max_clusters + 1):
            # Fit GMM
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm.fit(image_reshaped)

            # Predict cluster labels
            gmm_labels = gmm.predict(image_reshaped)

            # Compute silhouette score
            silhouette = silhouette_score(image_reshaped, gmm_labels)

            # Check if silhouette score is better than the current best
            if silhouette > best_silhouette_score:
                best_silhouette_score = silhouette
                best_gmm = gmm
                best_n_clusters = n_clusters

            # Update the progress bar
            pbar.update(1)

    # Generate segmented image using the best GMM model
    segmented_image = np.reshape(best_gmm.predict(image_reshaped), converted_img.shape[:2])

    return best_gmm, segmented_image, best_n_clusters



def calculate_cluster_percentages(u, total_pixels):
    """
    Calculate the percentage of each cluster in the image based on fuzzy c-means membership values.

    Args:
        u (numpy.ndarray): Fuzzy c-means membership values array.
        total_pixels (int): Total number of pixels in the image.

    Returns:
        dict: Dictionary containing the percentage of each cluster.
    """
    total_pixels = u.shape[1]

    # Calculate the sum of membership values for each cluster
    cluster_sums = np.sum(u, axis=1)

    # Calculate the percentage of each cluster
    cluster_percentages = (cluster_sums / total_pixels) * 100

    return {f'Cluster {i + 1}': percentage for i, percentage in enumerate(cluster_percentages)}


def calculate_cluster_pixel_counts(u):
    """
    Calculate the number of pixels in each cluster based on fuzzy c-means membership values.

    Args:
        u (numpy.ndarray): Fuzzy c-means membership values array.

    Returns:
        dict: Dictionary containing the count of pixels in each cluster.
    """
    # Calculate the count of pixels for each cluster
    cluster_counts = np.sum(u, axis=1)

    return {f'Cluster {i + 1}': count for i, count in enumerate(cluster_counts)}




def plotter(target, val, bins_center, hist, save_path=None):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    for i in tqdm(range(4), desc='Plotting Progress'):
        if i == 0:
            axes[0, 0].imshow(target, cmap='gray')
            axes[0, 0].set_title("Original Image")

        elif i == 1:
            axes[0, 1].imshow(target, cmap='gray')
            axes[0, 1].set_title("Thresholded Image")

        elif i == 2:
            axes[1, 0].plot(bins_center, hist, lw=2)
            axes[1, 0].set_title("Histogram")

        elif i == 3:
            axes[1, 1].imshow(target < val, cmap='gray')
            axes[1, 1].set_title("Thresholded Image")

    if save_path:
        # Ensure the directory exists before saving
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = parse_arg()
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("Created output folder:", output_folder)

    # Searching for filenames
    filenames = []
    for root, directories, files in os.walk(input_folder):
        for fn in files:
            basename, ext = fn.rsplit('.', 1)
            if ext.lower() in ['jpeg', 'jpg', 'png', 'tif']:
                filenames.append(os.path.join(root, fn))

    print("File #:", len(filenames))

    # Process each file
    for count, fin in tqdm(enumerate(filenames), total=len(filenames), desc='Processing Images'):
        # Initialize percentages dictionary
        # percentages = {}

        # Perform image processing operations
        filename = os.path.splitext(os.path.basename(fin))[0]  # Get the filename without extension
        img = imread_func(fin, True)
        img = im2double_func(img)
        converted_img = change_colorspace(img, True, output_folder, filename, args.colorspace)

        # clustered_img, labels, centers = kmeans_clustering(converted_img, 2, 'random', False, 'img', n_init_value='auto')



        # perform the GMM segmentation
        best_gmm, segmented_image, best_n_clusters = perform_gmm_segmentation(converted_img)
        # Print the best number of clusters
        print("Best Number of Clusters:", best_n_clusters)


        # Call the function to find the optimal number of clusters using silhouette score
        # optimal_clusters = find_optimal_clusters_silhouette(np.reshape(clustered_img, (-1, clustered_img.shape[-1])), max_clusters=10)

        # Using the fuzzy c-means clustering function and extracting first two clusters
        #cluster_images, cntr = fuzzy_cmans_automatic_Th(converted_img, optimal_clusters, False, filename)
        img_cluster1 = segmented_image[0] if len(segmented_image) > 0 else None
        img_cluster2 = segmented_image[1] if len(segmented_image) > 1 else None

        if len(segmented_image) > 0:
            total_pixels = img.shape[0] * img.shape[1]
            total = 0
            for i in range(0, 10):
                percentage = np.sum(segmented_image[i]) / total_pixels
                percentage_int = int(round(percentage * 100))  # Convert to integer percentage
                total += percentage_int  # Add integer percentage to total
                print(str(percentage_int))  # Print integer percentage

                # Export percentages to CSV file
                csv_filename = os.path.join(output_folder, "cluster_percentages.csv")
                with open(csv_filename, 'a', newline='') as csvfile:
                    fieldnames = ['Filename'] + list(percentage_int.keys())  # Include 'Filename' as a fieldname
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if count == 0:
                        writer.writeheader()
                    writer.writerow({'Filename': filename, **percentage_int})  # Include the filename in the row


        # Plotting (if you are still using these plots)
        if img_cluster1 is not None:
            target = img_cluster1
            val = filters.threshold_otsu(target)
            hist, bins_center = exposure.histogram(target)
            plotter(target, val, bins_center, hist, save_path=os.path.join(output_folder, f"plot_{filename}_cluster1.png"))
        
        if img_cluster2 is not None:
            target = img_cluster2
            val = filters.threshold_otsu(target)
            hist, bins_center = exposure.histogram(target)
            plotter(target, val, bins_center, hist, save_path=os.path.join(output_folder, f"plot_{filename}_cluster2.png"))

    # update_progress("Progress...{}/{}".format(count + 1, len(filenames)), (count + 1) / len(filenames))

print()
print("DONE!")
 
    
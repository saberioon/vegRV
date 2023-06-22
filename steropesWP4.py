"""
SteropesWP4.py\n
Author: Mohammadmhedi Saberioon
email: mohammadmehdi.saberioon@ilvo.vlanderen.be \n
Description: This code is developed for the WP4 of the Steropes project.
The main goal of this code is to segment the vegetation from the RGB images. \n 
Date: 2023-06-22  \n
"""




import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
from skimage import filters
from skimage import exposure
import argparse
import skfuzzy as fuzz
#from sklearn.cluster import DBSCAN


# -------------- These two libraris are for showing the progress------------------
import sys
import time
import os
#import json
import csv
# --------------for logging all messages ---------------
import datetime

VERSION = "1.0.1"
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

    for i in range(img.shape[0]):
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


def kmeans_clustering(img, n_of_clusters, init_value, to_save_or_not, output_save_name):
  """
    Perform k-means clustering on the input image.

    Args:
        img (numpy.ndarray): Input image array.
        n_of_clusters (int): Number of clusters to create.
        init_value (str): Initialization method for cluster centers. Supported values: 'random', 'k-means++'.
        to_save_or_not (bool): Flag indicating whether to save the clustered image.
        output_save_name (str): Output save name for the clustered image.

    Returns:
        numpy.ndarray: The clustered image array.

    Raises:
        ValueError: If an unsupported init_value is provided.

    Notes:
        - Supported values for init_value: 'random' or 'k-means++' (default).
        - The clustered image is saved if to_save_or_not is True.
    """
  log_message("kmeans clustering ...")
  kmeans = KMeans(n_clusters=n_of_clusters, init = init_value).fit(np.reshape(img,(-1,img.shape[2])))
  clustered_img = np.reshape(kmeans.predict(np.reshape(img,(-1,img.shape[2]))), img.shape[:-1])

  if to_save_or_not:
    filename = output_save_name + '_clustered.jpg'
    cv2.imwrite(filename, clustered_img[::-1,::-1])

  log_message("kmeans clustering completed")
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
                  The first element is the cluster 1 image, the second element is the cluster 2 image,
                  and the third element is the cluster centers array.
    """

    log_message("fuzzy cmeans clustering ...")
    img_shape = img.shape
    if len(img_shape) < 3:
      img = np.expand_dims(img, axis=2)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.reshape(img,(-1,img.shape[2])), n_of_clusters, 2, error=0.005, maxiter=1000, init=None)
    cluster1 = np.reshape(cntr[0,:], img.shape[:-1])
    cluster2 = np.reshape(cntr[1,:], img.shape[:-1])

    cluster1_img = np.zeros((img.shape[0], img.shape[1]))
    cluster2_img = np.zeros((img.shape[0], img.shape[1]))

    val1 = filters.threshold_otsu(cluster1)
    cluster1_img[cluster1 < val1]=1

    val2 = filters.threshold_otsu(cluster2)
    cluster2_img[cluster2 < val2]=1

    if to_save_or_not:
      filename = output_save_name + '_cluster1.jpg'
      cv2.imwrite(filename, cluster1_img[::-1,::-1])

      filename = output_save_name + '_cluster2.jpg'
      cv2.imwrite(filename, cluster2_img[::-1,::-1])
    log_message("fuzzy cmeans clustering completed")
    return cluster1_img, cluster2_img, cntr




def calculate_cluster_percentages(cluster1_img, cluster2_img):
    """
    Calculate the percentage of each cluster in the image.

    Args:
        cluster1_img (numpy.ndarray): Image of cluster 1.
        cluster2_img (numpy.ndarray): Image of cluster 2.

    Returns:
        dict: Dictionary containing the percentage of each cluster.
    """
    total_pixels = cluster1_img.size + cluster2_img.size
    cluster1_percentage = (cluster1_img.sum() / total_pixels) * 100
    cluster2_percentage = (cluster2_img.sum() / total_pixels) * 100

    return {
        'Cluster 1': cluster1_percentage,
        'Cluster 2': cluster2_percentage
    }

# ------------------------------------------------plotting 

def plotter(target, val, bins_center, hist):
    log_message("Plotting ... ")
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.imshow(target, cmap='jet', interpolation='nearest')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(target < val, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(133)
    plt.plot(bins_center, hist, lw=2)
    plt.axvline(val, color='k', ls='--')

    plt.tight_layout()
    #plt.show()
    log_message("Plotting completed")   



# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = parse_arg()
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("Created output folder:", output_folder)  # Print the path of the output folder

    # Searching for filenames
    filenames = []
    for root, directories, files in os.walk(input_folder):
        for fn in files:
            basename, ext = fn.rsplit('.', 1)
            if ext.lower() in ['jpeg', 'jpg', 'png', 'tif']:
                filenames.append(os.path.join(root, fn))

    print("File #:", len(filenames))

    # Rest of the code
    for count, fin in enumerate(filenames):
        # Perform image processing operations
        filename = os.path.splitext(os.path.basename(fin))[0]  # Get the filename without extension
        img = imread_func(fin, True)
        img = im2double_func(img)
        # hsv, hls, yiq = change_colorspace(img, True, output_folder, os.path.splitext(os.path.basename(fin))[0])
        converted_img = change_colorspace(img, True, output_folder, os.path.splitext(os.path.basename(fin))[0], args.colorspace)
        clustered_img = kmeans_clustering(converted_img, 2, 'random', False, 'img')

        img_cluster1, img_cluster2, cntr = fuzzy_cmans_automatic_Th(clustered_img, 2, False, args.colorspace)

        # Calculate cluster percentages
        percentages = calculate_cluster_percentages(img_cluster1, img_cluster2)

        # Export percentages to CSV file
        csv_filename = os.path.join(output_folder, "cluster_percentages.csv")
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['Filename'] + list(percentages.keys())  # Include 'Filename' as a fieldname
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if count == 0:
                writer.writeheader()
            writer.writerow({'Filename': filename, **percentages})  # Include the filename in the row

        cluster1 = np.reshape(cntr[0, :], converted_img.shape[:-1])
        cluster2 = np.reshape(cntr[1, :], converted_img.shape[:-1])
        target = cluster1
        val = filters.threshold_otsu(target)
        hist, bins_center = exposure.histogram(target)
        plotter(target, val, bins_center, hist)

        # Save the plots
        plt.savefig(os.path.join(output_folder, "plot_{}.png".format(count)))

        update_progress("Progress...{}/{}".format(count + 1, len(filenames)), (count + 1) / len(filenames))

    print()
    print("DONE!")







#  if __name__ == "__main__":
#     parser = parse_arg()
#     args = parser.parse_args()

#     input_folder = args.input
#     output_folder = args.output


#     # Create output folder if it does not exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print("Created output folder:", output_folder)  # Print the path of the output folder


#     # Searching for filenames
#     filenames = []
#     for root, directories, files in os.walk(input_folder):
#         for fn in files:
#             basename, ext = fn.rsplit('.', 1)
#             if ext.lower() in ['jpeg', 'jpg', 'png', 'tif']:
#                 filenames.append(os.path.join(root, fn))

#     print("File #:", len(filenames))

#     # Rest of the code
#     for count, fin in enumerate(filenames):
#         # Perform image processing operations
#         filename = os.path.splitext(os.path.basename(fin))[0]  # Get the filename without extension
#         img = imread_func(fin, True)
#         img = im2double_func(img)
#         #hsv, hls, yiq = change_colorspace(img, True, output_folder, os.path.splitext(os.path.basename(fin))[0])
#         converted_img = change_colorspace(img, True, output_folder, os.path.splitext(os.path.basename(fin))[0], args.colorspace)
#         clustered_img = kmeans_clustering(converted_img, 2 , 'random', False, 'img')
        
#         img_cluster1, img_cluster2, cntr = fuzzy_cmans_automatic_Th(clustered_img, 2, False, args.colorspace)
#         #cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.reshape(img,(-1,img.shape[2])), 2, 2, error=0.005, maxiter=1000, init=None)
        
#         # Calculate the cluster percentages
#         # cluster1_percentage = calculate_cluster_percentage(img_cluster1)
#         # cluster2_percentage = calculate_cluster_percentage(img_cluster2)

#         # Calculate cluster percentages
#         percentages = calculate_cluster_percentages(img_cluster1, img_cluster2)

#         # # Export percentages to CSV file
#         # csv_filename = os.path.join(output_folder, "cluster_percentages.csv")
#         # with open(csv_filename, 'w', newline='') as csvfile:
#         #    fieldnames = ['Filename'] + list(percentages.keys())  # Include 'Filename' as a fieldname
#         #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         #    writer.writeheader()
#         #    writer.writerow({'Filename': filename, **percentages})  # Include the filename in the row


#         csv_filename = os.path.join(output_folder, "cluster_percentages.csv")

#         # Create the CSV file and write the header row
#         with open(csv_filename, 'w', newline='') as csvfile:
#             fieldnames = ['Filename'] + list(percentages.keys())  # Include 'Filename' as a fieldname
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()

#             # Loop over the image files and write the cluster percentages to the CSV file
#             for filename, percentages in results.items():
#                 writer.writerow({'Filename': filename, **percentages})  # Include the filename in the row


#         # Print the cluster percentages
#         # print("Cluster 1 Percentage:", cluster1_percentage)
#         # print("Cluster 2 Percentage:", cluster2_percentage)
        
#         cluster1 = np.reshape(cntr[0,:], converted_img.shape[:-1])
#         cluster2 = np.reshape(cntr[1,:], converted_img.shape[:-1]) 
#         target = cluster1
#         val = filters.threshold_otsu(target)
#         hist, bins_center = exposure.histogram(target)
#         plotter(target, val, bins_center, hist)

#         # Save the plots
#         plt.savefig(os.path.join(output_folder, "plot_{}.png".format(count)))


#         update_progress("Progress...{}/{}".format(count + 1, len(filenames)), (count + 1) / len(filenames))


#     print()
#     print("DONE!")








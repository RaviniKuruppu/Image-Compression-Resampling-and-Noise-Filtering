# noise filter

import cv2
import numpy as np

# mean noise filter
def mean_filter(image,ksize):
  height, width = len(image), len(image[0])
  filter_image = np.zeros((height, width, 3), dtype=np.uint8)
  # mean kernel
  kernel = np.ones((ksize,ksize))
  kernel /= kernel.sum()
  #print(kernel)

  output_height = height - ksize + 1
  output_width = width - ksize + 1

  for i in range(output_height):
        for j in range(output_width):
            filter_image[i][j] = sum(
                image[i + m][j + n] * kernel[m][n]
                for m in range(ksize)
                for n in range(ksize)
            )
  return filter_image

# median noise filter
def median_filter(image, kernel_size):
    height, width = len(image), len(image[0])
    ksize = kernel_size // 2
    filtered_image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(ksize, height - ksize):
        for x in range(ksize, width - ksize):
            neighborhood = [
                image[i][j]
                for i in range(y - ksize, y + ksize + 1)
                for j in range(x - ksize, x + ksize + 1)
            ]

            # Sort the neighborhood
            sorted_neighborhood = sorted(neighborhood, key=lambda x: x[0])
            # find the median
            median_value = sorted_neighborhood[len(sorted_neighborhood) // 2]

            # Set the median value in the filtered image
            filtered_image[y][x] = median_value

    return filtered_image

#k-closest averaging filter
def k_closest_averaging_filter(image, k):
    height, width, _ = image.shape
    filter_image = np.zeros((height, width, 3), dtype=np.uint8)

    half_ksize = k // 2

    for i in range(height):
        for j in range(width):
            # Collect the k-closest pixels
            neighbors = []
            for m in range(max(0, i - half_ksize), min(height, i + half_ksize + 1)):
                for n in range(max(0, j - half_ksize), min(width, j + half_ksize + 1)):
                    neighbors.append(image[m][n])

            # Sort the neighbors based on their distances
            neighbors.sort(key=lambda pixel: np.sum((pixel - image[i][j]) ** 2))

            # Take the average of the k-closest pixels
            selected_neighbors = neighbors[:k]
            sum_values = [sum(channel) for channel in zip(*selected_neighbors)]
            mean_values = [sum_value // k for sum_value in sum_values]
            filter_image[i][j] = tuple(mean_values)

    return filter_image

#threshold averaging filter
def threshold_averaging_filter(image, ksize, threshold):
    height, width = len(image), len(image[0])
    filter_image = np.zeros((height, width, 3), dtype=np.uint8)

    kernel = np.ones((ksize, ksize))
    kernel /= kernel.sum()

    output_height = height - ksize + 1
    output_width = width - ksize + 1

    for i in range(output_height):
        for j in range(output_width):
            # Calculate the mean value
            mean_value = sum(
                image[i + m][j + n] * kernel[m][n]
                for m in range(ksize)
                for n in range(ksize)
            )

            # Apply thresholding
            if (image[i][j]-mean_value).any() > threshold:
                filter_image[i][j] = mean_value
            else:
                filter_image[i][j] = image[i][j]

    return filter_image


ksize = 5
#read the image
image = cv2.imread('/content/taj-rgb-noise.jpg')

#filterImage=mean_filter(image,ksize)
#filterImage=median_filter(image,ksize)
filterImage=k_closest_averaging_filter(image,5)
#filterImage=threshold_averaging_filter(image, ksize, 10)

cv2.imwrite('filterImage.jpg', filterImage)

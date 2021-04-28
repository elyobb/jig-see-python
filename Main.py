import cv2
import numpy as np
from matplotlib import pyplot as plt


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


img = cv2.imread("img.jpg", 0)
img50 = cv2.imread("img50.jpg", 0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img50, 127, 255, cv2.THRESH_BINARY)

images = [img, thresh1, img50, thresh2]
titles = ["Image w/ 10pc", "Processed Image", "Image w/ 50pc", "Processed Image"]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(images[i], connectivity=8)
    sizes = stats[1:, -1]
    if (i == 0 or i == 2 ):
        plt.title(titles[i])
    else:
        # for every component in the image, you keep it only if it's above min_size
        realNum = 0
        avg = sum(sizes) / len(sizes)
        sizes = sizes.tolist()
        sizes.remove(max(sizes))
        print(sizes.sort())
        notOutliers = reject_outliers(np.array(sizes))
        onlyOutliers = list(set(sizes) - set(notOutliers))

        plt.title('# Pieces Identified: ' + str(len(onlyOutliers)))
    plt.xticks([]), plt.yticks([])

plt.show()

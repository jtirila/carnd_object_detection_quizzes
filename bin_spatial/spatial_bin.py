import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
image = mpimg.imread('cutout1.jpg')


# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    if not color_space == 'RGB':
        assert color_space in ("HLS", "HSV", "LUV", "GRAY")
        colorspace_val = getattr(cv2, "COLOR_RGB2{}".format(color_space))
        # Convert image to new color space (if specified)
        # Use cv2.resize().ravel() to create the feature vector
        img_cc = cv2.cvtColor(img, colorspace_val)
    else:
        img_cc = img

    small_img = cv2.resize(img_cc, size)
    features = small_img.ravel()  # Remove this line!
    # Return the feature vector
    return features


feature_vec = bin_spatial(image, color_space='RGB', size=(32, 32))

# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
plt.show()
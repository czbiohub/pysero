"""
12to16bit.py
------------
This file takes 12bit images and converts them to 16bit images. It writes them to a specified target directory.
By: Marcus Forst
"""
#%% imports
import pandas as pd
import skimage
from skimage import io
from skimage import exposure
import os
import cv2
import numpy as np
#%%
SOURCE = '/Volumes/GoogleDrive/My Drive/ELISAarrayReader/images_nautilus/2020-08-14-COVID_Aug14_OJ_Plate11_2020-08-14 19-29-59.049679/0_renamed'
TARGET = '/Volumes/GoogleDrive/My Drive/ELISAarrayReader/images_nautilus/2020-08-14-COVID_Aug14_OJ_Plate11_2020-08-14 19-29-59.049679/0_renamed_16bit'
#filemap="C:\\Users\\gt8ma\\OneDrive\\Documents\\2019-2020 School Year\\BioHub\\2020-05-01-17-29-54-COVID_May1_JBassay_images\\cuttlefish_wellToFile_windows.xlsx"
ROTATION_ANGLE = 0

"""
This function loops through the files in the source directory, adjusts them from 12 to 16 bits, and writes
them to the target directory.
"""
def main():
    # Go to source directory
    filenames = os.listdir(SOURCE)
    # Sort the files, imperfect but better than nothing.
    filenames.sort()
    os.makedirs(TARGET, exist_ok=True)
    # Loop through files
    for filename in filenames:
        # Only loop through image files
        if not ".png" in filename:
            continue
        print(filename)
        # Join the filepath for the source directory
        src_path = os.path.join(SOURCE, filename)
        # Read in the file
        img = io.imread(src_path)
        # Rotate by 180 degrees
        img = np.rot90(img, k=2)
        # Rescale the intensity
        img = exposure.rescale_intensity(img, in_range='uint12')
        # Switch directories and add the filename.
        dst_path = os.path.join(TARGET, filename)
        # Save the file
        cv2.imwrite(dst_path, img)
        print("---------")


# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == '__main__':
    main()
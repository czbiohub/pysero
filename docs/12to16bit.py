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

SOURCE = "G:\\.shortcut-targets-by-id\\1FHidlgj8aczP41QFyhJt6NPhLPeOLmo7\\ELISAarrayReader\\images_nautilus\\2020-06-24-COVID_June24_OJAssay_Plate9_images_655_2020-06-24 18-26-58.173049\\0 renamed"
TARGET = "G:\\.shortcut-targets-by-id\\1FHidlgj8aczP41QFyhJt6NPhLPeOLmo7\\ELISAarrayReader\\images_nautilus\\2020-06-24-COVID_June24_OJAssay_Plate9_images_655_2020-06-24 18-26-58.173049\\0 renamed 16bit"
#filemap="C:\\Users\\gt8ma\\OneDrive\\Documents\\2019-2020 School Year\\BioHub\\2020-05-01-17-29-54-COVID_May1_JBassay_images\\cuttlefish_wellToFile_windows.xlsx"
ROTATION_ANGLE = 0

"""
This function loops through the files in the source directory, adjusts them from 12 to 16 bits, and writes
them to the target directory.
"""
def main():
    # Go to source directory
    files = os.listdir(SOURCE)
    # Sort the files, imperfect but better than nothing.
    files.sort()
    # Loop through files
    for filename in files:
        # Only loop through image files
        if not ".png" in file:
            continue
        print(filename)
        # Join the filepath for the source directory
        file = os.path.join(SOURCE, filename)
        # Read in the file
        img = io.imread(file)
        # Rescale the intensity
        img = exposure.rescale_intensity(img, in_range='uint12')
        # Switch directories and add the filename.
        new_file = os.path.join(TARGET, file)
        # Save the file
        io.imsave(new_file, img)
        print("---------")


# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == '__main__':
    main()
"""
rename_only.py
--------------
from Bryant updated by Marcus Forst

"""

import shutil, os
# path to source data folder
source = "G:\\.shortcut-targets-by-id\\1FHidlgj8aczP41QFyhJt6NPhLPeOLmo7\\ELISAarrayReader\\images_nautilus\\2020-06-24-COVID_June24_OJAssay_Plate9_images_655_2020-06-24 18-20-58.782514\\0"
# path to target data folder
target = "G:\\.shortcut-targets-by-id\\1FHidlgj8aczP41QFyhJt6NPhLPeOLmo7\\ELISAarrayReader\\images_nautilus\\2020-06-24-COVID_June24_OJAssay_Plate9_images_655_2020-06-24 18-20-58.782514\\0 renamed"
files = os.listdir(source)
letters = ['A','B','C','D','E','F','G','H']

for file in files:
    if not '.png' in file:
        continue
    file_list = file.strip()
    file_list = file.split('_')
    row = file_list[0]
    col = file_list[1]
    t_row = letters[int(row)]
    t_col = int(col)+1
    print(t_row, t_col)
    shutil.copyfile(source+f'/{file}', target+f"/{t_row}{t_col}.png")




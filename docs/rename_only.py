"""
rename_only.py
--------------
from Bryant updated by Marcus Forst

"""

import shutil, os
# path to source data folder
source = '/Volumes/GoogleDrive/My Drive/ELISAarrayReader/images_nautilus/2020-08-14-COVID_Aug14_OJ_2020-08-14 19-29-59.049679/0'
# path to target data folder
target = '/Volumes/GoogleDrive/My Drive/ELISAarrayReader/images_nautilus/2020-08-14-COVID_Aug14_OJ_2020-08-14 19-29-59.049679/0_renamed'
files = os.listdir(source)
letters = ['A','B','C','D','E','F','G','H']
os.makedirs(target, exist_ok=True)
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




#!/usr/bin/env python
# coding: utf-8

# %% Setup


from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import skimage.io as io
from natsort import natsorted
import seaborn as sns;

sns.set_context("talk")
font = {'size': 10, 'weight': 'normal', 'family': 'arial'}
matplotlib.rc('font', **font)


# %% Set paths

# %% First path
data_folder1 = r'/Users/janie.byrum/Desktop/comet segmentation test/For figure/pysero_biotin_fiducial_20200921_1451'
metadata_path1 = os.path.join(data_folder1, 'pysero_output_data_metadata.xlsx')
OD_path1 = os.path.join(data_folder1, 'median_ODs.xlsx')
int_path1 = os.path.join(data_folder1, 'median_intensities.xlsx')
bg_path1 = os.path.join(data_folder1, 'median_backgrounds.xlsx')
#scienion1_path = os.path.join(data_folder1, '2020-06-04-16-08-27-COVID_June4_JBassay_analysis.xlsx')
statsperwell_path = os.path.join(data_folder1, 'stats_per_well 1.xlsx')

# %% Read intensity sum from pysero
stats_xlsx = pd.ExcelFile(statsperwell_path)
sheets = stats_xlsx.sheet_names

statsperwell_df = pd.concat([pd.read_excel(stats_xlsx, sheet_name=s)
                .assign(sheet_name=s) for s in sheets])

# %% Plots

# I'm trying to plot just one antigen from one well initially

well_list = ['A1']
antigen = ['SARS CoV2 RBD 500']
A1data_df = statsperwell_df[(statsperwell_df['sheet_name'].isin(well_list)) & (statsperwell_df['antigen'].isin(antigen))]

from itertools import combinations
for index in list(combinations(A1data_df.index,2)):
    subdf = (A1data_df.loc[index,:])
    if subdf.isin({'comet_status':[0]}):
        paircomet = ['yes']
        subdf['Pair Comet']=paircomet
    else:
        paircomet =['no']
        subdf['Pair Comet'] = paircomet


fig_path = os.path.join(data_folder1, 'comet plots')
os.makedirs(fig_path, exist_ok=True)

well_list = ['A1']
antigen = ['SARS CoV2 RBD 500']
markers = 'o'
hue = 'comet_status'

A1data_df = statsperwell_df[(statsperwell_df['sheet_name'].isin(well_list)) & (statsperwell_df['antigen'].isin(antigen))]
palette = sns.color_palette()

g = sns.scatterplot(x="intensity_sum", y="od_norm", hue=hue, palette=palette, data=A1data_df)


    # for antigen, ax in zip(antigens, g.axes.flat):
    #     df_fit = sub_python_df_fit[(sub_python_df_fit['antigen'] == antigen)]
    #     sub_serum_df = sub_df[(sub_df['antigen'] == antigen)]
    #     palette = sns.color_palette(n_colors=len(sub_serum_df[hue].unique()))
    #
    #     sns.lineplot(x="serum dilution2", y="OD", hue=hue, hue_order=sera_fit_list, data=df_fit,
    #                  style=style, palette=palette2,
    #                  ax=ax, legend=False)

        # sns.scatterplot(x="serum dilution2", y="OD", hue=hue, estimator='mean', data=sub_serum_df, palette=palette, markers=markers, ax=ax, legend=False)
        # ax.set(xlim=[10e-5, 10e0])
        # ax.set(xscale="log")




        # ax.set(ylim=[-0.05, 1.5])
plt.savefig(os.path.join(fig_path, '{}_{}_{}_fit.jpg'.format('cometplottest', antigen, well_list)),dpi=300, bbox_inches='tight')

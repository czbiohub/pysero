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


def antigen2D_to_df1D(xlsx_path, sheet, data_col):
    """
    Convert old 2D output format (per antigen) to 1D dataframe
    :param xlsx_path:
    :param sheet:
    :param data_col:
    :return:
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet, index_col=0)
    df = df.unstack().reset_index(name=data_col)  # unpivot (linearize) the table
    df.rename(columns={'level_1': 'antigen_row', 'level_0': 'antigen_col'}, inplace=True)
    df[['antigen_row', 'antigen_col']] = df[['antigen_row', 'antigen_col']].applymap(int)
    df = df[['antigen_row', 'antigen_col', data_col]]
    df.dropna(inplace=True)
    return df


def well2D_to_df1D(xlsx_path, sheet, data_col):
    """
    Convert new 2D output format (per well) to 1D dataframe
    :param xlsx_path:
    :param sheet:
    :param data_col:
    :return:
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet, index_col=0)
    df = df.unstack().reset_index(name=data_col)  # unpivot (linearize) the table
    df.rename(columns={'level_1': 'row_id', 'level_0': 'col_id'}, inplace=True)
    df['well_id'] = df.row_id + df.col_id.map(str)
    df = df[['well_id', data_col]]
    return df


# %% Set paths

# %% First path
data_folder1 = r'/Users/janie.byrum/Desktop/comet segmentation test/For figure/pysero_biotin_fiducial_20200921_1451'
metadata_path1 = os.path.join(data_folder1, 'pysero_output_data_metadata.xlsx')
OD_path1 = os.path.join(data_folder1, 'median_ODs.xlsx')
int_path1 = os.path.join(data_folder1, 'median_intensities.xlsx')
bg_path1 = os.path.join(data_folder1, 'median_backgrounds.xlsx')
#scienion1_path = os.path.join(data_folder1, '2020-06-04-16-08-27-COVID_June4_JBassay_analysis.xlsx')
statsperwell_path = os.path.join(data_folder1, 'stats_per_well 1.xlsx')




# %% Read antigen and plate info
sheet_names = ['serum ID','serum ID1',
               'serum cat',
               'serum dilution','serum dilution2',
               'serum type',
               'secondary ID',
               'secondary dilution']
plate_info_df = pd.DataFrame()
with pd.ExcelFile(metadata_path1) as metadata_xlsx:
    # get sheet names that are available in metadata
    sheet_names = list(set(metadata_xlsx.sheet_names).intersection(sheet_names))
    for sheet_name in sheet_names:
        sheet_df = pd.read_excel(metadata_path1, sheet_name=sheet_name, index_col=0)
        sheet_df = sheet_df.unstack().reset_index(name=sheet_name)  # unpivot (linearize) the table
        sheet_df.rename(columns={'level_1': 'row_id', 'level_0': 'col_id'}, inplace=True)
        if plate_info_df.empty:
            plate_info_df = sheet_df
        else:
            plate_info_df = pd.merge(plate_info_df,
                                     sheet_df,
                                     how='left', on=['row_id', 'col_id'])
plate_info_df['well_id'] = plate_info_df.row_id + plate_info_df.col_id.map(str)
sheet_names.append('well_id')
# convert to number and non-numeric to NaN
plate_info_df['serum dilution'] = \
    plate_info_df['serum dilution'].apply(pd.to_numeric, errors='coerce')
plate_info_df.dropna(inplace=True)
# %%
if np.all(plate_info_df['serum dilution'] >= 1):
    # convert dilution to concentration
    plate_info_df['serum dilution'] = 1 / plate_info_df['serum dilution']
plate_info_df.drop(['row_id', 'col_id'], axis=1, inplace=True)

# %% Read antigen information.
antigen_df = antigen2D_to_df1D(xlsx_path=metadata_path1, sheet='antigen_array', data_col='antigen')

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

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
import seaborn as sns

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
cometod_path = os.path.join(data_folder1, 'manual comet od df.xlsx')

# %% Second path
data_folder2 = r'/Users/janie.byrum/Desktop/comet segmentation test/For figure/pysero_IgG fiducial_20201006_0942'
metadata_path2 = os.path.join(data_folder2, 'pysero_output_data_metadata.xlsx')
OD_path2 = os.path.join(data_folder2, 'median_ODs.xlsx')
int_path2 = os.path.join(data_folder2, 'median_intensities.xlsx')
bg_path2 = os.path.join(data_folder2, 'median_backgrounds.xlsx')
#scienion1_path = os.path.join(data_folder1, '2020-06-04-16-08-27-COVID_June4_JBassay_analysis.xlsx')
statsperwell_path2 = os.path.join(data_folder2, 'stats_per_well1.xlsx')
# %% Read antigen and plate info

# %% Read intensity sum from pysero
stats1_xlsx = pd.ExcelFile(statsperwell_path)
sheets1 = stats1_xlsx.sheet_names

stats2_xlsx = pd.ExcelFile(statsperwell_path2)
sheets2 = stats2_xlsx.sheet_names

statsperwell1_df = pd.concat([pd.read_excel(stats1_xlsx, sheet_name=s)
                .assign(sheet_name=s) for s in sheets1])

statsperwell2_df = pd.concat([pd.read_excel(stats2_xlsx, sheet_name=s)
                .assign(sheet_name=s) for s in sheets2])

statsperwelldfs = [statsperwell1_df, statsperwell2_df]

statsperwell_df = pd.concat(statsperwelldfs)
# %% Read in manual comet df
cometod_xlsx = pd.ExcelFile(cometod_path)

cometod_df = cometod_xlsx.parse("Sheet1")
# %% Merge manual comet df and od_norm data

for index,row in cometod_df.iterrows():
    tempwell1_df = statsperwell_df[
        (statsperwell_df["grid_row"] == row["grid_row"]) & (statsperwell_df["grid_col"] == row["grid_col"]) & (
        statsperwell_df["sheet_name"] == row["well1"])]
    cometod_df.at[index,'OD1'] = tempwell1_df['od_norm']
    cometod_df.at[index,'comet_status1'] = tempwell1_df['comet_status']
    tempwell2_df = statsperwell_df[
        (statsperwell_df["grid_row"] == row["grid_row"]) & (statsperwell_df["grid_col"] == row["grid_col"]) & (
        statsperwell_df["sheet_name"] == row["well2"])]
    cometod_df.at[index,'OD2'] = tempwell2_df['od_norm']
    cometod_df.at[index,'comet_status2'] = tempwell2_df['comet_status']
cometod_df['comet_statuspair'] = cometod_df['comet_status1'] + cometod_df['comet_status2']

# %% Plots

# Plot a scatter plot with OD1 on x axis and OD2 on y axis with hue = comet_statuspair

fig_path = os.path.join(data_folder1, 'comet plots')
os.makedirs(fig_path, exist_ok=True)

spots = cometod_df['spot'].unique()
sera = cometod_df['serum ID'].unique()
no_of_cometvals = cometod_df['comet_statuspair'].unique()
print(no_of_cometvals)
# # cometvals = [255,510,0,nan]
hue = "comet_statuspair"
palette = sns.color_palette("bright", n_colors=3)

g=sns.relplot(x="OD1", y="OD2", palette=palette, hue=hue, data= cometod_df)
g.set(xlim=(0, 0.45), ylim=(0, 0.45))



# g = sns.FacetGrid(cometod_df, col="comet_statuspair")
# g.map_dataframe(sns.scatterplot, x="OD1", y="OD2", hue=hue, data= cometod_df)
# g.set_axis_labels("OD spot 1", "OD spot 2")
# g.add_legend()
# palette = sns.color_palette('Greens',n_colors=len(cometvals)

# sns.lmplot(x="OD1", y="OD2", hue=hue, palette=palette, data= cometod_df)

# sns.scatterplot(x="OD1", y="OD2", hue=hue, palette=palette, data=cometod_df)

# markers = 'o'
# hue = 'comet_status'
#
# A1data_df = statsperwell_df[(statsperwell_df['sheet_name'].isin(well_list)) & (statsperwell_df['antigen'].isin(antigen))]
# palette = sns.color_palette()
#
# g = sns.scatterplot(x="intensity_sum", y="od_norm", hue=hue, palette=palette, data=A1data_df)


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
plt.savefig(os.path.join(fig_path, '{}_{}_{}_fit.jpg'.format('comet_statuspair', 'with legend', 'split')),dpi=300, bbox_inches='tight')


# %% Separate plots

g = sns.FacetGrid(cometod_df, col="comet_statuspair", margin_titles=True)
g.map_dataframe(sns.scatterplot, x="OD1", y="OD2")
g.set_axis_labels("OD1", "OD2")
# g.set_titles(col_template="{col_name} patrons", row_template="{row_name}")
for ax, title in zip(g.axes.flat, ['both comets', 'one comet', 'no comets']):
    ax.set_title(title)
g.set(xlim=(0, 0.45), ylim=(0, 0.45))



plt.savefig(os.path.join(fig_path, '{}_{}_{}_fit.jpg'.format('comet_statuspair', 'all', 'split')),dpi=300, bbox_inches='tight')

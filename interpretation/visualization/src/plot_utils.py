#!/usr/bin/env python3

import numpy as np
import pandas as pd

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = 'plotly_white'

from tqdm import tqdm
from scipy.optimize import curve_fit
import natsort
import os
import re


class PyseroLoader(object):

    """
    Loads serum and antigen metadata from pysero output xlsx and provides
    pandas dataframes for analysis and plotting
    """

    def __init__(self, data_dir, scienion_xlsx):

        self.data_dir = data_dir

        self.metadata_path = os.path.join(
            self.data_dir, 'pysero_output_data_metadata.xlsx'
        )

        self.OD_path = os.path.join(
            self.data_dir, 'median_ODs_per_well.xlsx'
        )

        self.scienion_path = os.path.join(
            self.data_dir, scienion_xlsx
        )

        self.validate_paths()

        self.well_regex = re.compile(
            "[ABCDEFGH][0123456789]+$"
        )


        # Meta Data
        self.SerumMeta = pd.DataFrame()
        self.AntigenMeta = pd.DataFrame()

        # Optical Density Data
        self.Pysero_OD = pd.DataFrame()
        self.Scienion_OD = pd.DataFrame()

        # Processed Dataframes
        self.Pysero_Frame = pd.DataFrame()
        self.Scienion_Frame = pd.DataFrame()

    def validate_paths(self):
        """
        Confirms all necessary files are within an existing directory
        """

        assert os.path.isdir(self.data_dir)
        assert os.path.isfile(self.metadata_path)
        assert os.path.isfile(self.OD_path)
        assert os.path.isfile(self.scienion_path)

    def Load_Pysero_OD(self):

        """
        Load well data for each spot in pysero output
        """

        xls = pd.ExcelFile(self.OD_path)
        well_od = []

        # iter wells in sheets
        for well in xls.sheet_names:

            # only accept wells mathing regex
            if not self.well_regex.match(well):
                continue

            # load matrix
            od_matrix = pd.read_excel(xls, well)

            # convert to longform
            od_longform = od_matrix.melt(id_vars = od_matrix.columns[0])

            # annotate columns and append well position
            od_longform.columns = ['a_row', 'a_col', 'OD']
            od_longform['well'] = well

            # grow list
            well_od.append(od_longform)

        self.Pysero_OD = pd.concat(well_od)

    def Load_Scienion_OD(self, one_index=True):

        """
        Load well data for each spot in scienion output

        Transforms scienion 255 bounded inverse intensity into one bounded OD
        """

        xls = pd.ExcelFile(self.scienion_path)
        well_od = []

        columns_to_keep = [
            'a_row', 'a_col', 'OD', 'well'
        ]

        for well in xls.sheet_names:

            if not self.well_regex.match(well):
                continue

            intensity_frame = pd.read_excel(xls, well)
            intensity_frame[['a_row', 'a_col']] = intensity_frame.\
                apply(
                    lambda x : [int(i) for i in x.ID.split('-')[-2:]],
                    axis = 1, result_type = 'expand'
                )

            # Converts from one-indexed to zero-indexed antigen matrix
            if one_index:
                intensity_frame['a_row'] -= 1
                intensity_frame['a_col'] -= 1


            # transform intensity and background
            ## Swap 255-bounding to 1-bounding and invert value
            transform_arr = lambda x : 1 - (x / 255)

            intensity_frame['intensity'] = transform_arr(
                intensity_frame['Median'].values
                )

            intensity_frame['background'] = transform_arr(
                intensity_frame['Background Median'].values
                )

            intensity_frame['OD'] = np.log10(
                intensity_frame['background'] / intensity_frame['intensity']
                )

            intensity_frame['well'] = well

            well_od.append(intensity_frame[columns_to_keep])

        self.Scienion_OD = pd.concat(well_od)

    def FlattenMatrix(self, mat, mat_type = 'ag', var_name = 'variable'):

        """
        Flatten from long to wide with consistent naming
        """

        melted = mat.melt(
            id_vars = mat.columns[0],
        )

        if mat_type == 'ag' :
            melted.columns = ['a_row', 'a_col', var_name]
        else:
            melted.columns = ['s_row', 's_col', var_name]

        return melted

    def LoadAntigenMeta(self, xls):

        """
        Pull antigen meta data from relevant sheets
        """

        def label_duplicates(frame, columns):

            """
            label duplicate values in a dataframe,
            used to catch same antigen~dilution replicates
            """
            tuple_list = [tuple(i) for i in frame[columns].values]

            duplicate_order = []
            d = {}
            for t in tuple_list:
                if t not in d:
                    d[t] = 0
                duplicate_order.append(d[t])
                d[t] += 1

            return np.array(duplicate_order)



        antigen_id = self.FlattenMatrix(
            pd.read_excel(xls, "antigen_array"),
            var_name = "AntigenID"
        )

        antigen_type = self.FlattenMatrix(
            pd.read_excel(xls, "antigen_type"),
            var_name = "AntigenType"
        )

        self.AntigenMeta = antigen_id.\
            merge(antigen_type).\
            dropna()

        # label antigen replicates
        self.AntigenMeta['AntigenReplicate'] = label_duplicates(
            self.AntigenMeta, ['AntigenID', 'AntigenType']
            )

    def LoadSerumMeta(self, xls):

        """
        Pull serum meta data from relevant sheets
        """

        serum_id = self.FlattenMatrix(
            pd.read_excel(xls, "serum ID"),
            mat_type = 'serum', var_name = "SerumID"
        )

        serum_type = self.FlattenMatrix(
            pd.read_excel(xls, "serum type"),
            mat_type = 'serum', var_name = "SerumType"
        )

        serum_dilution = self.FlattenMatrix(
            pd.read_excel(xls, "serum dilution"),
            mat_type = 'serum', var_name = "SerumDilution"
        )

        secondary_dilution = self.FlattenMatrix(
            pd.read_excel(xls, "secondary dilution"),
            mat_type = 'serum', var_name = 'SecondaryDilution'
        )


        self.SerumMeta = serum_id.\
            merge(serum_type).\
            merge(serum_dilution).\
            merge(secondary_dilution)

        self.SerumMeta['well'] = self.SerumMeta.s_row + self.SerumMeta.s_col.astype(str)

    def LoadMeta(self):

        """
        Load antigen metadata
        """

        xls = pd.ExcelFile(self.metadata_path)

        self.LoadAntigenMeta(xls)
        self.LoadSerumMeta(xls)

    def LoadOD(self):

        """
        Load optical density matrices as dataframes
        """

        self.Load_Pysero_OD()
        self.Load_Scienion_OD()

    def FullPyseroOutput(self):

        """
        Returns a Merged metadata with pysero OD dataframe
        """

        if self.SerumMeta.empty:
            self.LoadMeta()

        if self.Pysero_OD.empty:
            self.LoadOD()

        if self.Pysero_Frame.empty:
            self.Pysero_Frame = self.Pysero_OD.\
                merge(self.SerumMeta).\
                merge(self.AntigenMeta)

            self.Pysero_Frame['Pipeline'] = 'Pysero'

        return self.Pysero_Frame

    def FullScienionOutput(self):

        """
        Returns a Merged metadata with pysero OD dataframe
        """

        if self.SerumMeta.empty:
            self.LoadMeta()

        if self.Scienion_OD.empty:
            self.LoadOD()

        if self.Scienion_Frame.empty:
            self.Scienion_Frame = self.Scienion_OD.\
                merge(self.SerumMeta).\
                merge(self.AntigenMeta)
            self.Scienion_Frame['Pipeline'] = 'Scienion'

        return self.Scienion_Frame


class SerumDilutionPlotter(object):

    """
    Class for fitting and plotting 4 parameter logarithmic function to OD data
    """

    def __init__(self):

        self.color_dict = {
            'positive' : '#a23b22',
            'negative' : '#0a87be'
        }

        self.style_dict = {
            'positive' : None,
            'negative' : 'dash'
        }


        pass

    def fourPL(self, x, A, B, C, D):

        """
        4 parameter logarithmic curve
        """

        return ((A-D)/(1.0+((x/C)**(B))) + D)

    def fit(self, frame):

        """
        fits a 4 parameter logarithmic curve to each subgrouping of the data,
        returns a tidy dataframe of fit values
        """

        grouping_vals = ['SerumType', 'SerumID', 'SecondaryDilution', 'AntigenID']
        frames = []

        for idx, df in tqdm(frame.groupby(grouping_vals), desc = 'Fitting curves over groupings...'):

            if np.unique(df.SerumDilution.values).size == 1:
                continue

            SerumType, SerumID, SecondaryDilution, AntigenID = idx

            prefit = pd.DataFrame({
                'x' : df.SerumDilution.astype(float).values,
                'y' : df.OD.astype(float).values
                }).\
                groupby('x').\
                apply(
                    lambda x : pd.Series({
                        'y' : x.y.mean()
                    })
                ).\
                reset_index()
            prefit['fit_type'] = 'pre'

            params = [0, 1, 5e-4, 1]

            params, params_covariance = curve_fit(
                self.fourPL, prefit.x.values, prefit.y.values, params,
                bounds=(0, np.inf), maxfev=1e5
            )

            logspace_x = np.logspace(
                np.log10(prefit.x.min()),
                np.log10(prefit.x.max()),
                50
                )

            postfit = pd.DataFrame({
                'x' : logspace_x,
                'y' : self.fourPL(logspace_x, *params),
                'fit_type' : 'post',
                'SerumType' : SerumType,
                'SerumID' : SerumID,
                'SecondaryDilution' : SecondaryDilution,
                'AntigenID' : AntigenID
            })

            frames.append(postfit)


        full_set = pd.concat(frames)

        return full_set

    def build_trace(self, idx, subframe, ag_id_idx):
        """
        creates a trace for a given subframe
        """


        serum_type, serum_id, antigen_id = idx

        trace = go.Scatter(
            x = subframe.x, y = subframe.y,
            line = dict(
                color = self.color_dict[serum_type],
                dash = self.style_dict[serum_type]
                ),
            name = serum_id,
            legendgroup = serum_id,
            showlegend = (ag_id_idx == 1)
        )

        return trace

    def plot_antigen(self, frame, secondary_dilution, antigen_id, height = None):

        """
        Plot 4 parameter fit logarithmic curves for a given antigen at a
        given secondary dilution
        """

        frame = frame[
            (frame.AntigenID.str.contains(antigen_id)) & \
            (frame.SecondaryDilution == secondary_dilution)
        ]

        ag_dilutions = natsort.natsorted(frame.AntigenID.unique())

        fig = make_subplots(
            rows = len(ag_dilutions), cols = 1,
            subplot_titles=ag_dilutions,
            shared_yaxes=True, shared_xaxes=True
        )

        for idx, df in frame.groupby(['SerumType', 'SerumID', 'AntigenID']):

            serum_type, serum_id, antigen_id = idx
            ag_id_idx = ag_dilutions.index(antigen_id) + 1

            trace = self.build_trace(idx, df, ag_id_idx)
            fig.append_trace(trace, row = ag_id_idx, col = 1)


        if not height:
            height = 400 * (len(ag_dilutions) + 1)

        fig.update_layout(
            height = height,
            title_text = "Secondary Dilution : {}".format(secondary_dilution)
            )
        fig.update_yaxes(range = (0, frame.y.max() + 0.1))

        return fig

    def write_plot(self, fig, path, format = 'html'):

        """
        Write plot to file
        """

        expected_formats = ['html', 'pdf', 'svg', 'png', 'jpeg']

        assert format in expected_formats

        if format == 'html':
            pio.write_html(fig, path)
        else:
            pio.write_image(fig, path)

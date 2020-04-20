import cv2 as cv
from datetime import datetime
import glob
import numpy as np
import os
import pandas as pd
import re
import time

import array_analyzer.extract.image_parser as image_parser
import array_analyzer.extract.img_processing as img_processing
import array_analyzer.extract.txt_parser as txt_parser
import array_analyzer.load.debug_images as debug_plots
import array_analyzer.transform.point_registration as registration
import array_analyzer.transform.array_generation as array_gen

FIDUCIALS = [(0, 0), (0, 1), (0, 5), (7, 0), (7, 5)]
FIDUCIALS_IDX = [0, 5, 6, 30, 35]
FIDUCIALS_IDX_8COLS = [0, 7, 8, 40, 47]
SCENION_SPOT_DIST = 82
# The expected standard deviations could be estimated from training data
# Just winging it for now
STDS = np.array([500, 500, .1, .001])  # x, y, angle, scale


def point_registration(input_folder, output_folder, debug=False):
    """
    For each image in input directory, detect spots using particle filtering
    to register fiducial spots to blobs detected in the image.

    :param str input_folder: Input directory containing images and an xml file
        with parameters
    :param str output_folder: Directory where output is written to
    :param bool debug: For saving debug plots
    """
    xml_path = glob.glob(input_folder + '/*.xml')
    if len(xml_path) > 1 or not xml_path:
        raise IOError("Did not find unique xml")
    xml_path = xml_path[0]

    # parsing .xml
    fiduc, spots, repl, params = txt_parser.create_xml_dict(xml_path)

    # creating our arrays
    spot_ids = txt_parser.create_array(params['rows'], params['columns'])
    antigen_array = txt_parser.create_array(params['rows'], params['columns'])

    # adding .xml info to these arrays
    spot_ids = txt_parser.populate_array_id(spot_ids, spots)

    antigen_array = txt_parser.populate_array_antigen(antigen_array, spot_ids, repl)

    # save a sub path for this processing run
    run_path = os.path.join(
        output_folder,
        '_'.join([os.path.basename(os.path.normpath(input_folder)),
                  str(datetime.now().month),
                  str(datetime.now().day),
                  str(datetime.now().hour),
                  str(datetime.now().minute),
                  str(datetime.now().second)]),
    )
    os.makedirs(run_path, exist_ok=True)

    xlwriter_od = pd.ExcelWriter(os.path.join(run_path, 'ODs.xlsx'))
    pdantigen = pd.DataFrame(antigen_array)
    pdantigen.to_excel(xlwriter_od, sheet_name='antigens')
    if debug:
        xlwriter_int = pd.ExcelWriter(os.path.join(run_path, 'intensities.xlsx'))
        xlwriter_bg = pd.ExcelWriter(os.path.join(run_path, 'backgrounds.xlsx'))

    # ================
    # loop over images
    # ================
    images = [file for file in os.listdir(input_folder)
              if '.png' in file or '.tif' in file or '.jpg' in file]

    # remove any images that are not images of wells.
    well_images = [file for file in images if re.match(r'[A-P][0-9]{1,2}', file)]

    # sort by letter, then by number (with '10' coming AFTER '9')
    well_images.sort(key=lambda x: (x[0], int(x[1:-4])))

    for image_name in well_images:
        start_time = time.time()
        image = image_parser.read_gray_im(os.path.join(input_folder, image_name))

        props_array = txt_parser.create_array(
            params['rows'],
            params['columns'],
            dtype=object,
        )
        bgprops_array = txt_parser.create_array(
            params['rows'],
            params['columns'],
            dtype=object,
        )

        nbr_grid_rows, nbr_grid_cols = props_array.shape
        fiducials_idx = FIDUCIALS_IDX
        if nbr_grid_cols == 8:
            fiducials_idx = FIDUCIALS_IDX_8COLS

        spot_coords = img_processing.get_spot_coords(
            image,
            min_area=200,
            min_thresh=0,
            max_thresh=255,
            min_circularity=0,
            min_convexity=0,
            min_dist_between_blobs=10,
            min_repeatability=2,
        )

        # Initial estimate of spot center
        mean_point = tuple(np.mean(spot_coords, axis=0))
        grid_coords = registration.create_reference_grid(
            mean_point=mean_point,
            nbr_grid_rows=nbr_grid_rows,
            nbr_grid_cols=nbr_grid_cols,
            spot_dist=SCENION_SPOT_DIST,
        )
        fiducial_coords = grid_coords[fiducials_idx, :]

        particles = registration.create_gaussian_particles(
            mean_point=(0, 0),
            stds=STDS,
            scale_mean=1.,
            angle_mean=0.,
            nbr_particles=1000,
        )
        # Optimize estimated coordinates with iterative closest point
        t_matrix = registration.particle_filter(
            fiducial_coords=fiducial_coords,
            spot_coords=spot_coords,
            particles=particles,
            stds=STDS,
            stop_criteria=0.1
        )
        # Transform grid coordinates
        reg_coords = np.squeeze(cv.transform(np.array([grid_coords]), t_matrix))

        # Crop image
        im_crop, crop_coords = img_processing.crop_image_from_coords(
            im=image,
            grid_coords=reg_coords
        )
        im_crop = im_crop / np.iinfo(im_crop.dtype).max
        # # Estimate and remove background
        background = img_processing.get_background(
            im_crop,
            fit_order=2,
            normalize=False,
        )

        props_placed_by_loc, bgprops_by_loc = array_gen.get_spot_intensity(
            coords=crop_coords,
            im_int=im_crop,
            background=background,
            params=params,
        )
        props_array_placed = image_parser.assign_props_to_array(
            props_array,
            props_placed_by_loc,
        )
        bgprops_array = image_parser.assign_props_to_array(
            bgprops_array,
            bgprops_by_loc,
        )
        od_well, int_well, bg_well = image_parser.compute_od(
            props_array_placed,
            bgprops_array,
        )

        # pd_od = pd.DataFrame(od_well)
        # pd_od.to_excel(xlwriter_od, sheet_name=image_name[:-4])

        print("Time to register grid to {}: {:.3f} s".format(
            image_name,
            time.time() - start_time),
        )

        # ==================================
        # SAVE FOR DEBUGGING
        if debug:
            start_time = time.time()
            # Save spot and background intensities.
            # pd_int = pd.DataFrame(int_well)
            # pd_int.to_excel(xlwriter_int, sheet_name=image_name[:-4])
            # pd_bg = pd.DataFrame(bg_well)
            # pd_bg.to_excel(xlwriter_bg, sheet_name=image_name[:-4])

            output_name = os.path.join(run_path, image_name[:-4])
            # # Evaluate accuracy of background estimation with green (image), magenta (background) overlay.
            # im_bg_overlay = np.stack([background, im_crop, background], axis=2)
            # io.imsave(output_name + "_crop_bg_overlay.png",
            #           (255 * im_bg_overlay).astype('uint8'))

            # Save a composite of all spots, where spots are from source or from region prop
            debug_plots.plot_od(
                od_well,
                int_well,
                bg_well,
                output_name,
            )
            debug_plots.save_composite_spots(
                im_crop,
                props_array_placed,
                output_name,
                from_source=True,
            )
            debug_plots.plot_registration(
                image,
                spot_coords,
                grid_coords[fiducials_idx, :],
                reg_coords,
                output_name,
            )

            print("Time to save debug images: {:.3f} s".format(
                time.time() - start_time),
            )

            # xlwriter_int.close()
            # xlwriter_bg.close()

    xlwriter_od.close()
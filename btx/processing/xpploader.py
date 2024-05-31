from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import tables

def get_imgs_thresh(run_number, experiment_number, roi,
                    energy_filter=[8.8, 5], i0_threshold=200,
                    ipm_pos_filter=[0.2, 0.5], time_bin=2,
                    time_tool=[0., 0.005]):
    global exp, h5dir
    exp = experiment_number
    h5dir = Path(f'/sdf/data/lcls/ds/xpp/{exp}/hdf5/smalldata/')

    rr = SMD_Loader(run_number)

    idx_tile = rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][0, 0]
    mask = rr.UserDataCfg.jungfrau1M.mask[idx_tile][rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1, 0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1, 1],
                                                    rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2, 0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2, 1]]

    I0 = rr.ipm2.sum[:]
    laser_on_mask = np.array(rr.evr.code_90) == 1
    laser_off_mask = np.array(rr.evr.code_91) == 1

    E0, dE = energy_filter
    thresh_1, thresh_2 = E0 - dE, E0 + dE
    thresh_3, thresh_4 = 2 * E0 - dE, 2 * E0 + dE
    thresh_5, thresh_6 = 3 * E0 - dE, 3 * E0 + dE

    imgs = rr.jungfrau1M.ROI_0_area[:, roi[0]:roi[1], roi[2]:roi[3]]
    imgs_cleaned = imgs.copy()
    imgs_cleaned[(imgs_cleaned < thresh_1)
                 | ((imgs_cleaned > thresh_2) & (imgs_cleaned < thresh_3))
                 | ((imgs_cleaned > thresh_4) & (imgs_cleaned < thresh_5))
                 | (imgs_cleaned > thresh_6)] = 0
    
    imgs_thresh = imgs_cleaned#* mask[roi[0]:roi[1], roi[2]:roi[3]]

    tt_arg = time_tool[0]
    laser_delays = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS) * tt_arg

    # Add TimeTool filtering
    arg_tt_amplitude = np.array(rr.tt.AMPL) > time_tool[1]

    # Update laser_on_mask and laser_off_mask based on TimeTool conditions
    if tt_arg == 1:
        laser_on_mask = laser_on_mask & arg_tt_amplitude
        laser_off_mask = laser_off_mask & arg_tt_amplitude

    return imgs_thresh, I0, laser_delays, laser_on_mask, laser_off_mask

def SMD_Loader(Run_Number: int) -> tables.File:
    """Load the Small Data"""
    fname = f'{exp}_Run{Run_Number:04d}.h5'
    fname = h5dir / fname
    return tables.open_file(fname).root


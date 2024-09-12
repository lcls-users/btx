import numpy as np
import argparse
import os
from mpi4py import MPI
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import holoviews as hv
from holoviews import opts
import pandas as pd
from btx.interfaces.ipsana import *
from btx.interfaces.ischeduler import JobScheduler

from psana import EventId
from Detector.UtilsEpix10ka import find_gain_mode
from Detector.UtilsEpix10ka import info_pixel_gain_mode_statistics_for_raw
from Detector.UtilsEpix10ka import map_pixel_gain_mode_for_raw

class RunDiagnostics:
    """
    Class to compute powders and trajectories of various image statistics
    and event characteristics for a given run.
    """    
    def __init__(self, exp, run, det_type, xtcdir=None, hutch=None):
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type,
                                  xtcdir=xtcdir, hutch=hutch, track_timestamps=True)
        self.pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(run))
        self.powders = dict() 
        self.stats = dict()
        self.gain_traj = None
        self.gain_map = None
        self.gain_mode = ''

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size() 

    def init_base_powders(self):
        """
        Initialize the base powders: sum, sum of squares, max, and min.
        Parameters
        ----------
        img_shape : tuple, 3d
            The shape of unassembled, calibrated images: (n_panels, n_x, n_y)
        """
        for key in ['sum', 'sqr', 'max', 'min']:
            self.powders[key] = np.zeros(self.psi.det.shape())

    def compute_base_powders(self, img):
        """
        Compute the base powders: max, sum, sum of squares.

        Parameters
        ----------
        img : numpy.ndarray, 3d
            unassembled, calibrated images of shape (n_panels, n_x, n_y)
        """
        if not self.powders:
            for key in ['sum', 'sqr', 'max', 'min']:
                self.powders[key] = img.copy()
        else:
            if img is not None:
                self.powders['sum'] += img
                self.powders['sqr'] += np.square(img)
                self.powders['max'] = np.maximum(self.powders['max'], img)
                self.powders['min'] = np.minimum(self.powders['min'], img)

    def finalize_powders(self):
        """
        Finalize powders calculation at end of the run, computing the
        max, avg, and std dev (and gain mode counts if applicable).
        """
        self.powders_final = dict()
        total_n_proc = self.comm.reduce(self.n_proc, MPI.SUM)
        powder_max = np.array(self.comm.gather(self.powders['max'], root=0))
        powder_min = np.array(self.comm.gather(self.powders['min'], root=0))
        powder_sum = np.array(self.comm.gather(self.powders['sum'], root=0))
        powder_sqr = np.array(self.comm.gather(self.powders['sqr'], root=0))
        if self.gain_map is not None:
            powder_gain = np.array(self.comm.gather(self.gain_map, root=0))

        if self.rank == 0:
            self.powders_final['max'] = np.max(powder_max, axis=0)
            self.powders_final['min'] = np.min(powder_min, axis=0)
            if total_n_proc > 0:
                self.powders_final['avg'] = np.sum(powder_sum, axis=0) / float(total_n_proc)
                self.powders_final['std'] = np.sqrt(np.sum(powder_sqr, axis=0) / float(total_n_proc) - np.square(self.powders_final['avg']))
            if self.gain_map is not None:
                self.powders_final['gain_mode_counts'] = np.sum(powder_gain, axis=0)
            if self.psi.det_type.lower() != 'rayonix':
                for key in self.powders_final.keys():
                    self.powders_final[key] = assemble_image_stack_batch(self.powders_final[key], self.pixel_index_map)
                self.panel_mask = assemble_image_stack_batch(np.ones(self.psi.det.shape()), self.pixel_index_map)
 
    def save_powders(self, outdir, raw_img=False):
        """
        Save powders to output directory.

        Parameters
        ----------
        output : str
            path to directory in which to save powders, optional
        raw_img : bool
            if True, save powder files with _raw nomenclature
        """
        if self.rank == 0:
            suffix=""
            if raw_img:
                suffix = "_raw"
            for key in self.powders_final.keys():
                np.save(os.path.join(outdir, f"r{self.psi.run:04}_{key}{suffix}.npy"), self.powders_final[key])

    def save_traces(self, outdir, raw_img=False):
        """
        Save trajectories of statistics to output directory.
        
        Parameters
        ----------
        output : str
            path to directory in which to save traces, optional
        raw_img : bool
            if True, save powder files with _raw nomenclature
        """
        if self.rank == 0:
            suffix=""
            if raw_img:
                suffix = "_raw"
            for key in self.stats_final.keys():
                np.save(os.path.join(outdir, f"r{self.psi.run:04}_trace_{key}{suffix}.npy"), self.stats_final[key])

    def load_traces(self, outdir, raw_img=False):
        """
        Load previously saved trajectories of statistics.

        Parameters
        ----------
        outdir: str
            path to directory where traces have been saved
        raw_img : bool
            if True, load trace files with _raw nomenclature
        """
        self.stats_final = dict()
        if self.rank == 0:
            suffix=""
            if raw_img:
                suffix = "_raw"
            for key in ['mean', 'std', 'max', 'min', 'beam_energy_eV', 'photon_energy_eV']:
                filepath_key = os.path.join(outdir, f"r{self.psi.run:04}_trace_{key}{suffix}.npy")
                try:
                    self.stats_final[key] = np.load(filepath_key)
                except FileNotFoundError as e:
                    print(f"Error while attempting to read {filepath_key}")

    def compute_stats(self, evt, img):
        """
        Compute the following image stats: mean, std deviation, max, min.

        Parameters
        ----------
        evt : psana.Event object
            individual psana event       
        img : numpy.ndarray, 3d
            unassembled, calibrated images of shape (n_panels, n_x, n_y)
        """
        if not self.stats:
            for key in ['mean', 'std', 'max', 'min', 'beam_energy_eV', 'total_intensity', 'photon_energy_eV']:
                self.stats[key] = np.zeros(self.psi.max_events - self.psi.counter)
        
        beam_energy_mJ   = self.psi.get_fee_gas_detector_energy_mJ_evt(evt)
        photon_energy_eV = self.psi.get_photon_energy_eV_evt(evt)                
        if beam_energy_mJ is not None and photon_energy_eV is not None:
            try:
                beam_energy_eV = beam_energy_mJ / 1.602e-16 * self.psi.get_beam_transmission()
                self.stats['beam_energy_eV'][self.n_proc] = beam_energy_eV
            except NotImplementedError as e:
                print(f"Beam energy not known for current hutch.\n",
                      "Speak with a beamline scientist to find the proper PV")
            self.stats['photon_energy_eV'][self.n_proc] = photon_energy_eV

            if img is None:
                self.stats['total_intensity'][self.n_proc] = np.nan
                self.stats['mean'][self.n_proc] = np.nan
                self.stats['std'][self.n_proc] = np.nan
                self.stats['max'][self.n_proc] = np.nan
                self.stats['min'][self.n_proc] = np.nan
            else:
                self.stats['total_intensity'][self.n_proc] = np.sum(img)/photon_energy_eV
                self.stats['mean'][self.n_proc] = np.mean(img)
                self.stats['std'][self.n_proc] = np.std(img)
                self.stats['max'][self.n_proc] = np.max(img)
                self.stats['min'][self.n_proc] = np.min(img)
        else:
            self.n_empty += 1
            self.stats['beam_energy_eV'][self.n_proc] = np.nan
            self.stats['photon_energy_eV'][self.n_proc] = np.nan
            self.stats['total_intensity'][self.n_proc] = np.nan
            self.stats['mean'][self.n_proc] = np.nan
            self.stats['std'][self.n_proc] = np.nan
            self.stats['max'][self.n_proc] = np.nan
            self.stats['min'][self.n_proc] = np.nan
                
    def finalize_stats(self):
        """
        Gather stats from various ranks into single arrays in self.stats_final.
        """
        self.stats_final = dict()
        for key in self.stats.keys():
            self.stats_final[key] = self.comm.gather(self.stats[key], root=0)

        self.stats_final['fiducials'] = self.comm.gather(np.array(self.psi.fiducials), root=0)
        if self.rank == 0:
            for key in self.stats_final.keys():
                self.stats_final[key] = np.hstack(self.stats_final[key])
                
        if self.gain_traj is not None:
            self.stats_final['gain_mode_counts'] = self.comm.gather(self.gain_traj, root=0)
            if self.rank == 0:
                self.stats_final['gain_mode_counts'] = np.hstack(self.stats_final['gain_mode_counts'])
                
    def get_gain_statistics(self, raw, gain_mode='AML-L'):
        """
        Retrieve statistics for a particular gain mode, specifically a map 
        of the number of times a pixel has been in a certain gain mode and 
        the number of pixels in that gain mode per event.

        Parameters
        ----------
        raw : ndarray, shape (det_shape)
            uncalibrated image
        gain_mode : str
            gain mode to retrieve statistics for, e.g. 'AML-L'
        """
        if self.gain_traj is None:
            self.gain_mode = gain_mode
            self.modes = {'FH':0, 'FM':1, 'FL':2, 'AHL-H':3, 'AML-M':4, 'AHL-L':5, 'AML-L':6}
            self.gain_traj = np.zeros(self.psi.max_events - self.psi.counter).astype(int)
            self.gain_map = np.zeros(self.psi.det.shape())

        stats = info_pixel_gain_mode_statistics_for_raw(self.psi.det, raw)
        self.gain_traj[self.n_proc] = int(stats.split(":")[1].split(",")[self.modes[gain_mode]])
        evt_map = map_pixel_gain_mode_for_raw(self.psi.det, raw)
        self.gain_map[evt_map==[self.modes[gain_mode]]] += 1
            
    def compute_run_stats(self, max_events=-1, mask=None, events_mask=None, powder_only=False, threshold=None, total_intensity=False, gain_mode=None, raw_img=False):
        """
        Compute powders and per-image statistics. If a mask is provided, it is 
        only applied to the stats trajectories, not in computing the powder.
        
        Parameters
        ----------
        max_events : int
            number of images to process; if -1, process entire run
        mask : str or np.ndarray, shape (n_panels, n_x, n_y)
            binary mask file or array in unassembled psana shape, optional 
        events_mask : np.ndarray, shape (n_events,)
            a Boolean mask of events considered for computing the statistics
        powder_only : bool
            if True, only compute the powder pattern
        threshold : float
            if supplied, exclude events whose mean is above this value
        total_intensity : bool
            if True, compute the total intensity
        gain_mode : str
            gain mode to retrieve statistics for, e.g. 'AML-L' 
        raw_img : bool
            if True, analyze raw rather than calibrated images
        """
        if mask is not None:
            if type(mask) == str:
                mask = np.load(mask) 
            assert mask.shape == self.psi.det.shape()

        self.psi.distribute_events(self.rank, self.size, max_events=max_events)
        start_idx, end_idx = self.psi.counter, self.psi.max_events
        self.n_proc, self.n_empty, n_empty_raw, n_excluded = 0, 0, 0, 0

        if self.psi.det_type.lower() == 'rayonix':
            if self.rank == 0:
                if self.check_first_evt(mask=mask):
                    print("First image of the run is an outlier and will be excluded")
                    start_idx += 1
                    
        self.init_base_powders()
        self.comm.Barrier()
        for idx in np.arange(start_idx, end_idx):
            if events_mask is not None and not events_mask[idx]:
                continue

            evt = self.psi.runner.event(self.psi.times[idx])
            self.psi.get_timestamp(evt.get(EventId))
            if raw_img:
                img = self.psi.det.raw(evt=evt)
                if self.psi.det_type == 'epix10k2M':
                    img = img & 0x3fff # exclude first two bits for powder calculation
            else:
                img = self.psi.det.calib(evt=evt)
            if img is None:
                self.n_empty += 1
                
            if threshold:
                if np.mean(img) > threshold:
                    print(f"Excluding event {idx} with image mean: {np.mean(img)}")
                    n_excluded += 1
                    img = None

            self.compute_base_powders(img)
            if not powder_only:
                if mask is not None:
                    img = np.ma.masked_array(img, 1-mask)
                self.compute_stats(evt, img)
                
            if gain_mode is not None and self.psi.det_type == 'epix10k2M':
                raw = self.psi.det.raw(evt)
                self.get_gain_statistics(raw, gain_mode)
            else:
                self.gain_mode = ''

            self.n_proc += 1
            if self.psi.counter == self.psi.max_events:
                break

        self.comm.Barrier()
        self.finalize_powders()
        if not powder_only:
            self.finalize_stats()
            print(f"Rank {self.rank}, no. empty images: {self.n_empty}, no. excluded images: {n_excluded}")

    def visualize_powder(self, tag='max', vmin=-1e5, vmax=1e5, output=None, figsize=12, dpi=300):
        """
        Visualize the powder image: the distribution of intensities as a histogram
        and the positive and negative-valued pixels on the assembled detector image.
        """
        if self.rank == 0:
            image = self.powders_final[tag]
        
            fig = plt.figure(figsize=(figsize,figsize),dpi=dpi)
            gs = fig.add_gridspec(2,2)

            irow=0
            ax1 = fig.add_subplot(gs[irow,:2])
            ax1.grid()
            ax1.hist(image.flatten(), bins=100, log=True, color='black')
            ax1.set_title(f'histogram of pixel intensities in powder {tag}', fontdict={'fontsize': 8})

            irow+=1
            ax2 = fig.add_subplot(gs[irow,0])
            im = ax2.imshow(np.where(image>0,0,image), cmap=plt.cm.gist_gray, 
                            norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=vmin, vmax=0.))
            ax2.axis('off')
            ax2.set_title(f'negative intensity pixels', fontdict={'fontsize': 6})
            plt.colorbar(im)

            ax3 = fig.add_subplot(gs[irow,1])
            im = ax3.imshow(np.where(image<0,0,image), cmap=plt.cm.gist_yarg, 
                            norm=colors.SymLogNorm(linthresh=1., linscale=1., vmin=0, vmax=vmax))
            ax3.axis('off')
            ax3.set_title(f'positive intensity pixels', fontdict={'fontsize': 6})
            plt.colorbar(im)

            if output is not None:
                plt.savefig(output)
        
    def visualize_stats(self, output=None):
        """
        Plot trajectories of run statistics.
        
        Parameters
        ----------
        output : str
            path for optionally saving plot to disk
        """
        if self.rank == 0:
            n_plots = len(self.stats_final.keys())-1
            keys = ['mean', 'max', 'min', 'std', 'beam_energy_eV', 'photon_energy_eV', 'gain_mode_counts']
            labels = ['mean(I)', 'max(I)', 'min(I)', 'std dev(I)', 
                      'Beam\nenergy (eV)', 'Photon\nenergy (eV)',
                      f'No. pixels in\n{self.gain_mode} mode']

            # Remove keys which are unavailable
            for key in keys:
                if key in self.stats_final.keys():
                    continue
                else:
                    idx = keys.index(key)
                    keys.pop(idx)
                    labels.pop(idx)
            
            f, axs = plt.subplots(n_plots, figsize=(n_plots*2.4,12), sharex=True)
            for n in range(n_plots):
                axs[n].plot(self.stats_final[keys[n]], c='black')
                axs[n].set_ylabel(labels[n], fontsize=12)       
            axs[n_plots-1].set_xlabel("Event index", fontsize=12)
            
            if output is not None:
                f.savefig(output, dpi=300)
                self.visualize_stats_hv_html(output,
                                             keys,
                                             color_by='beam_energy_eV')

    def visualize_stats_hv_html(self, output: str, keys: list, color_by: str):
        """! Create an \"interactive\" Holoviews plot embedded in HTML.

        Allows for selection of which statistics to display.

        @param output (str) Path to save the output HTML.
        @param keys (list) List of keys in `stats_final` to plot.
        @param color_by (str) Key to data to color the scatter plot points by.
        """
        import holoviews as hv
        import pandas as pd

        hv.extension('bokeh')
        plot_data = dict(self.stats_final)

        def plot_selector(df: pd.DataFrame, y: str, c: str) -> hv.Scatter:
            """! Create a scatter plot from a DataFrame.

            @param y (str) DataFrame key for y values.
            @param c (str) DataFrame key for coloring the scatter plot.
            @return scatter (hv.Scatter) Scatter plot.
            """
            scatter = hv.Scatter(df, kdims=['Event Id'], vdims=[y, c])
            return scatter.opts(tools=['hover'], width=800, color=c, framewise=True)

        def rebuild_dataframe(y: str, c: str):
            """! Create a new DataFrame from a new set of stats keys.

            Always creates a Series corresponding to the event axis.

            @param y (str) Stats key for the y-axis values in the DataFrame.
            @param c (str) Stats key for the color-by values in the DataFrame.
            @return df (pd.DataFrame) New pandas DataFrame.
            """
            num_evts: int = len(plot_data[y])
            df = pd.DataFrame({'Event Id' : np.arange(num_evts),
                               f'{y}' : plot_data[y],
                               f'{c}' : plot_data[c]})
            return df

        c = color_by
        plots_dict = { y:plot_selector(rebuild_dataframe(y, c), y, c) for y in keys }

        hmap = hv.HoloMap(plots_dict, kdims=['Y Value'])
        outfile = output[:-3] + 'html'
        hv.save(hmap, outfile)

    
    def visualize_gain_frequency(self, output=None):
        """
        Plot the distribution of the frequency with which each pixel
        appears in a particular gain mode.
        
        Parameters
        ----------
        output : str
            path for optionally saving plot to disk
        """
        if self.gain_map is None:
            print("Gain statistics were not retrieved.")
            return
            
        if self.rank == 0:
            f, ax1 = plt.subplots(figsize=(6,3.6))

            freq = self.powders_final['gain_mode_counts'][self.panel_mask==1]/len(self.stats_final['gain_mode_counts'])
            ax1.hist(freq, bins=50, color='black')
            ax1.set_yscale('log')

            ax1.set_ylabel("No. pixels", fontsize=12)
            ax1.set_xlabel(f"Frequency in {self.gain_mode} mode", fontsize=12)
            
            if output is not None:
                f.savefig(output, dpi=300, bbox_inches='tight')
                
    def visualize_energy_stats(self, output=None, outlier_threshold=-np.inf):
        """
        Plot the number of incoming photons per event, the number
        of incoming photons versus the mean image intensity, and 
        the beam energy versus photon energy. Energies are in eV.
        
        Parameters
        ----------
        output : str
            path for optionally saving plot to disk
        outlier_threshold : float
            mean intensity threshold for considering images outliers
        """
        if self.rank == 0:
            fig, axs = plt.subplots(nrows=1,ncols=3, figsize=(13,3.6))

            valid = [self.stats_final['mean']>outlier_threshold][0]

            n_incoming_photons = self.stats_final['beam_energy_eV']/self.stats_final['photon_energy_eV']
            scatter = axs[0].scatter(np.arange(len(n_incoming_photons))[valid], n_incoming_photons[valid], 
                                     s=2, c=self.stats_final['photon_energy_eV'][valid], cmap='magma')
            axs[0].scatter(np.arange(len(n_incoming_photons))[~valid], n_incoming_photons[~valid], 
                           s=2, c='grey', alpha=0.2, label='Outlier events')
            axs[0].set_xlabel('Event index', fontsize=12)
            axs[0].set_ylabel('No. incoming photons', fontsize=12)
            axs[0].grid()
            axs[0].legend(loc=1)

            cax0 = fig.add_axes([0.15,0.24,0.16,0.025])
            fig.colorbar(scatter, label='photon energy (eV)', 
                         cax=cax0, orientation='horizontal')

            hexbin = axs[1].hexbin(self.stats_final['mean'][valid], 
                                   n_incoming_photons[valid], 
                                   mincnt=1, gridsize=20, cmap='Reds')
            axs[1].set_xlabel('Mean image intensity', fontsize=12)
            axs[1].set_ylabel('No. incoming photons', fontsize=12)

            cax1 = fig.add_axes([0.43,0.24,0.16,0.025])
            fig.colorbar(hexbin, label='events per bin (outliers excluded)', 
                         cax=cax1, orientation='horizontal')

            hexbin = axs[2].hexbin(self.stats_final['photon_energy_eV'][valid],
                                   self.stats_final['beam_energy_eV'][valid], 
                                   mincnt=1, gridsize=20, cmap='Reds')
            axs[2].set_xlabel('photon energy (eV)', fontsize=12)
            axs[2].set_ylabel('beam energy (eV)', fontsize=12)

            cax2 = fig.add_axes([0.71,0.24,0.16,0.025])
            fig.colorbar(hexbin, label='events per bin (outliers excluded)', 
                         cax=cax2, orientation='horizontal')

            fig.subplots_adjust(wspace=0.3)
        
            if output is not None:
                fig.savefig(output, dpi=300, bbox_inches='tight')
    
    def visualize_stats_interactive(self, y_key='mean', color_by='beam_energy_eV'):
        """
        Plot trajectories of run statistics, interactively.

        Parameters
        ----------
        y_key : str
            self.stats_final key to plot
        color_by : str
            self.stats_final key used to color
        """
        hv.extension('bokeh')

        y = self.stats_final[y_key]

        df = pd.DataFrame({
            'evt_id' : np.arange(y.shape[0]), 
            'y' : y, 
            'color_by' : self.stats_final[color_by]})
        scatter = hv.Scatter(df, kdims=['evt_id'], vdims=['y', 'color_by'])
        return scatter.opts(opts.Scatter(tools=['hover'])).opts(width=800, color_index='color_by')

    def display_img_evt(self, event_id, 
            vmin=1, vmax=1e3, figsize=8, dpi=360, title=None, log=True, mask_negatives=True,
            method='pyplot'):
        """
        Display the detector image corresponding to a given event ID.

        Parameters
        ----------
        event_id : int
            Index of the desired event
        vmin : float
            Lowest pixel value
        figsize : int
            Figure size
        dpi : int
            Figure dots per inch
        title : str or None
            Figure title
        log : bool
            Whether the detector image is displayed on linear or log scale.
        mask_negatives: bool
            Whether negative intensity pixels should be masked.
        method: str
            options: 'pyplot' (default) or 'holoviews'
        """
        evt = self.psi.runner.event(self.psi.times[event_id])
        img = self.psi.det.image(evt=evt)
        if mask_negatives:
            img = np.ma.masked_where(img<=0,img)

        if method is 'pyplot':
            fig = plt.figure(figsize=(figsize,figsize),dpi=dpi)
            if log:
                plt.imshow(img, norm=LogNorm(vmin=vmin), interpolation='none')
            else:
                plt.imshow(img)
            plt.colorbar()
            if title is not None:
                plt.title(title)
            plt.xlabel('Y')
            plt.ylabel('X')
            plt.show()
        else:
            hv.extension('bokeh')
            hvimg = hv.Image(img)
            return hvimg.opts(width=800, height=800, logz=log, zlim=(vmin,vmax), 
                    colorbar=True, cmap='inferno', tools=['hover'])
        

    def check_first_evt(self, mask=None, scale_factor=5, n_images=5):
        """
        Check whether the first event of the run should be excluded; it's 
        considered an outlier if mean_0 > < mean_n + scale_factor * std_n >
        where n ranges from [1,n_images). If any of the first few events are
        None, exclude the first image to simplify matters.

        Parameters
        ----------
        mask : np.ndarray, shape (n_panels, n_x, n_y)
            binary mask file or array in unassembled psana shape, optional 
        scale_factor : float
            parameter that tunes  how conservative outlier rejection is
        n_images : int
            number of total images to compare, including first

        Returns:
        --------
        exclude : bool
            if True, first event was detected as an outlier
        """
        exclude = False
        means, stds = np.zeros(n_images), np.zeros(n_images)
        for cidx in range(n_images):
            evt = self.psi.runner.event(self.psi.times[cidx])
            img = self.psi.det.calib(evt=evt)
            if img is None:
                return exclude
            if mask is not None:
                img *= mask
            means[cidx], stds[cidx] = np.mean(img), np.std(img)
            
        if means[0] > np.mean(means[1:] + scale_factor * stds[1:]):
            exclude = True
        return exclude

class PixelTracker:

    """
    Class for tracking raw, calib, and (optionally) gain setting for a
    single pixel across a run.
    """
    
    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type, track_timestamps=True)
        self.pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(run))
        self.stats = dict()
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size() 
        
    def find_flattened_index(self, index):
        """
        From the input detector index, find the corresponding index in 
        the flattened detector array.

        Parameters
        ----------
        index : tuple, 2d
            pixel index on the assembled detector image

        Returns
        -------
        tuple, 3d
            corresponding index of the unassembled detector array
        """
        pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(self.psi.run))
        mapper = assemble_image_stack_batch(np.zeros(self.psi.det.shape()), pixel_index_map)
        mapper[index[0], index[1]] = -1
        mapper = disassemble_image_stack_batch(mapper, pixel_index_map)
        return tuple([elem[0] for elem in np.where(mapper == -1)])
    
    def finalize_stats(self, n_empty):
        """
        Gather stats from various ranks into single arrays in self.stats_final.
        
        Parameters
        ----------
        n_empty : int
            number of empty images in this rank
        """
        self.stats_final = dict()
        for key in self.stats.keys():
            if n_empty != 0:
                self.stats[key] = self.stats[key][:-n_empty]
            self.stats_final[key] = self.comm.gather(self.stats[key], root=0)
            if self.rank == 0:
                self.stats_final[key] = np.hstack(self.stats_final[key])
                
    def track_pixel(self, index, max_events=-1, gain_mode=True):
        """
        Compute powders and per-image statistics. If a mask is provided, it is 
        only applied to the stats trajectories, not in computing the powder.
        
        Parameters
        ----------
        index : tuple, 2d or 3d
            pixel index on the assembled or unassembled detector image
        max_events : int
            number of images to process; if -1, process entire run
        gain_mode : bool
            whether to track gain mode statistics
        """
        if len(index) == 2:
            index = self.find_flattened_index(index)
            print(f"Pixel index on disassembled detector: {index}")

        self.psi.distribute_events(self.rank, self.size, max_events=max_events)
        start_idx, end_idx = self.psi.counter, self.psi.max_events
        n_processed, n_empty = 0, 0
        
        for key in ['calib', 'raw']:
            self.stats[key] = np.zeros(end_idx - start_idx)
        if gain_mode:
            self.stats['gain'] = np.zeros(end_idx - start_idx).astype(int)

        for idx in np.arange(start_idx, end_idx):
            evt = self.psi.runner.event(self.psi.times[idx])
            raw = self.psi.det.raw(evt=evt)
            calib = self.psi.det.calib(evt=evt)
            
            if (calib is None) or (raw is None):
                n_empty += 1
                
            else:
                if self.psi.det_type == 'epix10k2M':
                    raw_bmask = raw & 0x3fff # exclude first two bits
                else:
                    raw_bmask = raw
                self.stats['raw'][n_processed] = raw_bmask[index[0], index[1], index[2]]
                self.stats['calib'][n_processed] = calib[index[0], index[1], index[2]]
                if gain_mode and self.psi.det_type == 'epix10k2M':
                    self.stats['gain'][n_processed] = map_pixel_gain_mode_for_raw(self.psi.det, raw)[index[0], index[1], index[2]]
            
            n_processed += 1
            if self.psi.counter + n_empty == self.psi.max_events:
                break
        
        self.index = index
        self.comm.Barrier()
        self.finalize_stats(n_empty)
        print(f"Rank {self.rank}, no. empty images: {n_empty}")
        
    def save_traj(self, outdir):
        """
        Save trajectories of raw, calib, and gain setting.
        
        Parameters:
        -----------
        outdir : str
            output directory
        """
        if self.rank == 0:
            for key in self.stats_final.keys():
                np.save(os.path.join(outdir, f"p_{self.index[0]}_{self.index[1]}_{self.index[2]}_{key}.npy"), 
                        self.stats_final[key])
                
    def visualize(self, outdir=None):
        """
        Plot the raw vs calibrated intensity, colored by gain setting.

        Parameters
        ----------
        outdir : str
            output directory
        """
        if self.rank == 0:
            f, ax1 = plt.subplots(figsize=(5,3.6))
            if 'gain' in self.stats_final.keys():
                color = self.stats_final['gain']
            else:
                color = 'black'
            ax1.scatter(self.stats_final['raw'], self.stats_final['calib'], c=color)
            ax1.set_xlabel("Raw intensity", fontsize=14)
            ax1.set_ylabel("Calibrated intensity", fontsize=14)
            ax1.set_title(f"Pixel ({self.index[0]}, {self.index[1]}, {self.index[2]})", fontsize=14)
        
            if outdir is not None:
                f.savefig(os.path.join(outdir, f"p_{self.index[0]}_{self.index[1]}_{self.index[2]}.png"),
                          bbox_inches='tight', dpi=300)

def main():
    """
    Perform run analysis, computing powders and tracking statistics.
    """
    params = parse_input()
    os.makedirs(os.path.join(params.outdir, "figs"), exist_ok=True)
    rd = RunDiagnostics(exp=params.exp,
                        run=params.run,
                        det_type=params.det_type,
                        xtcdir=params.xtcdir,
                        hutch=params.hutch)
    rd.compute_run_stats(max_events=params.max_events, 
                         mask=params.mask, 
                         threshold=params.mean_threshold,
                         total_intensity=params.total_intensity,
                         gain_mode=params.gain_mode,
                         raw_img=params.raw_img)
    rd.save_powders(params.outdir, raw_img=params.raw_img)
    rd.save_traces(params.outdir, raw_img=params.raw_img)
    suffix = ""
    if params.raw_img:
        suffix = "_raw"
    rd.visualize_powder(output=os.path.join(params.outdir, f"figs/powder_r{params.run:04}{suffix}.png"))
    rd.visualize_stats(output=os.path.join(params.outdir, f"figs/stats_r{params.run:04}{suffix}.png"))
    try:
        rd.visualize_energy_stats(output=os.path.join(params.outdir, f"figs/stats_energy_r{params.run:04}{suffix}.png"),
                                  outlier_threshold=params.outlier_threshold)
    except KeyError as e:
        print(f'PV : {e} unavailable. Energy stats not output')
    if params.gain_mode is not None:
        rd.visualize_gain_frequency(output=os.path.join(params.outdir, f"figs/gain_freq_r{params.run:04}{suffix}.png"))

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M',  required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory for powders and plots', required=True, type=str)
    parser.add_argument('-x', '--xtcdir', help='Alternative XTC directory', required=False, default=None,type=str)
    parser.add_argument('--hutch', help='Hutch name', required=False, default=None, type=str)
    parser.add_argument('-m', '--mask', help='Binary mask for computing trajectories', required=False, type=str)
    parser.add_argument('--max_events', help='Number of images to process, -1 for full run', required=False, default=-1, type=int)
    parser.add_argument('--mean_threshold', help='Exclude images with a mean above this threshold', required=False, type=float)
    parser.add_argument('--total_intensity', help='Compute total intensity of images', action='store_true')
    parser.add_argument('--gain_mode', help='Gain mode to track, e.g. AML-L for epix10k2M autoranging low gain', required=False, type=str)
    parser.add_argument('--raw_img', help="Analyze raw rather than calibrated images", action='store_true')
    parser.add_argument('--outlier_threshold', help='Consider images with a mean below this threshold outliers for energy stats plot',
                        required=False, default=-np.inf, type=float)

    return parser.parse_args()

if __name__ == '__main__':
    main()

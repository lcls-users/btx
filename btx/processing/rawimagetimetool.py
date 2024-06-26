import numpy as np
import matplotlib.pyplot as plt
import psana
from scipy.signal import fftconvolve
from scipy.interpolate import UnivariateSpline
import os

class RawImageTimeTool:
    """! Class to reimplement time tool analysis at LCLS from raw images.

    Uses psana to interface with experimental data to retrieve raw camera images.
    Edge detection is implemented in order to perform jitter correction.
    Refer to the documentation located at https://lcls-users.readthedocs.org/during/timetool
    for information on the theory and usage of this class.

    Properties:
    -----------
    model - Retrieve or load polynomial model coefficients.

    Methods:
    --------
    __init__(self, expmt: str, savedir: str) - Instantiate an analysis object for experiment expmt.
    open_run(self, run: str) - Open DataSource for a specified run.
    get_tt_name(self) - Determine time tool camera name in data files.
    calibrate(self, run: str, order: int, figs=True) - Calibrate the analysis object for jitter correction on specified run.
    process_run(self) - Perform edge detection on all events in a run.
    detect_edge(self, img: np.array, kernel: np.array) - Perform edge detection on an image.
    crop_image(self, img: np.array) - Crop time tool image to ROI.
    fit_calib(self, delays: np.array, edges: np.array, amplitudes: np.array,
            fwhm: np.array = None, order: int = 2) - Fit polynomial to calibration run data.
    ttstage_code(self, hutch) - Return the time tool target time stage for a given hutch.(
    edge_to_time(self, edge)
    actual_time_ps(self, edge, nominal)
    timetool_correct(self, edge, nominal, model, figs=True)
    plot_calib
    plot_hist
    """
    def __init__(self, expmt: str, savedir: str, debug: bool = False):
        """! Initializer for the time tool analysis object.

        @param expmt (str) Experiment accession name. E.g. cxilr1234.
        @param savedir (str) Directory where any output files are saved.
        """
        ## @var expmt
        # (str) Experiment accession name
        self.expmt = expmt

        ## @var hutch
        # (str) Experimental hutch. Extracted from expmt. Needed for camera/stage accession.
        self.hutch = self.expmt[:3]

        ## @var _model
        # (list) List of calibration model polynomial coefficients. Empty if not calibrated.
        self._model = []

        ## @var savedir
        # (str) Directory to save output files to. Figures may be placed in a subdirectory of this one.
        self.savedir = savedir

        ## @var debug
        # (bool) Whether to enable debug features e.g. early processing break.
        self.debug = debug

    def open_run(self, run: str):
        """! Open the psana DataSource and time tool detector for the specified
        run. This method MUST be called before calibration, analysis etc. The
        psana DataSource and Detector objects are stored in instance variables

        @param run (str) Run number (e.g. 5) or range of runs (e.g. 6-19).
        """

        ## @var ds
        # (psana.DataSource) DataSource object for accessing data for specified runs.
        self.ds = psana.MPIDataSource(f'exp={self.expmt}:run={run}')

        ## @var det
        # (psana.Detector) Detector object corresponding to the time tool camera.
        self.det = psana.Detector(self.get_tt_name(), self.ds.env())

    def get_tt_name(self) -> str:
        """! Run psana.MPIDataSource.detnames() to get associated detectors,
        process the output and find the name of the timetool detector.

        @return detname (str) Name of the timetool detector (full or alias).
        """
        for tup in self.ds.detnames():
            for name in tup:
                if any(alias in name.lower() for alias in ['opal', 'timetool']):
                    return name

    def calibrate(self, run: str, order: int = 2, figs: bool = True):
        """! Fit a calibration model to convert delay to time tool jitter
        correction. MUST be run before analysis if a model has not been previously
        fit. The fitted polynomial model is saved to a text file for later use.

        @param run (str) SINGLE run number for the experimental calibration run.
        @param order (int) Order of the polynomial to fit for calibration.
        @param figs (bool) Whether or not to print diagnostic figures.
        """
        self.open_run(run)

        delays, edges, ampls = self.process_run()
        self.fit_calib(delays, edges, ampls, None, order)

        run = self.format_run()
        outdir = f'{self.savedir}/calib'

        fname = f'{run}.out'
        self.write_file(self._model, fname, outdir)

        fname = f'EdgesFit_{run}.out'
        self.write_file(self.edges_fit, fname, outdir)

        fname = f'DelaysFit_{run}.out'
        self.write_file(self.delays_fit, fname, outdir)

        if figs:
            self.plot_calib(delays, edges, self._model)
            self.plot_hist(self.edges_fit)

    def format_run(self) -> str:
        """! Format the run(s) number for output file names. The format is
        r0000 for a single run and r0000-r0000 for a range of runs.
        """
        run = self.ds.ds_string.split(':')[1].split('=')[1]
        if '-' in run:
            run = run.split('-')
            start = run[0]
            end = run[1]

            run = f'r{int(start):04}-{int(end):04}'
        else:
            run = f'r{int(run):04}'
        return run

    def process_run(self, calib=True) -> (np.array, np.array, np.array):
        """! Perform edge detection for all images in a run.

        @param calib (bool) Whether or not this is a calibration run.

        @return delays (np.array) Array of delays used.
        @return edges (np.array) Detected edge position.
        @return ampls (np.array) Convolution amplitudes for the detected edges.
        @return stamps (np.array) Unique event identifier stamps. NOT returned in calibration.
        """
        kernel = np.zeros([200])
        kernel[:100] = 1

        delays = []
        edges = []
        ampls = []
        fwhms = []

        if not calib:
            stamps = []

        stage_code = self.ttstage_code(self.hutch)

        for idx, evt in enumerate(self.ds.events()):
            if idx > 10000 and self.debug:
                break
            
            delays.append(self.ds.env().epicsStore().value(stage_code))
            try:
                img = self.det.image(evt=evt)
                edge, ampl, fwhm = self.detect_edge(img, kernel)

                edges.append(edge)
                ampls.append(ampl)
                fwhms.append(fwhm)

                if not calib:
                    evtid = evt.get(psana.EventId)
                    evtfid = evtid.fiducials()
                    evttime = evtid.time()
                    stamp = f'{evttime[0]}-{evttime[1]}-{evtfid}'
                    stamps.append(stamp)

            except Exception as e:
                delays.pop(-1)
        if calib:
            return np.array(delays), np.array(edges), np.array(ampls)
        else:
            return np.array(delays), np.array(edges), np.array(ampls), np.array(stamps)

    def detect_edge(self, img: np.array, kernel: np.array) -> (float, int, float):
        """! Detects an edge in an image projection using convolution with a
        kernel function.

        @param img (array-like) 2D image array.
        @param kernel (array-like) 2

        @return edge (int) Pixel (column) of the detected edge position.
        @return ampl (float) Convolution amplitude. Can be used for filtering.
        @return fwhm (float) Full-width half-max of convolution. Can be used for filtering.
        """
        cropped = self.crop_image(img)

        proj = np.sum(cropped, axis=0)
        proj -= np.min(proj)
        proj /= np.max(proj)

        trace = fftconvolve(proj, kernel, mode='same')
        edge = trace.argmax()
        ampl = trace[edge]
        fwhm = self.measure_fwhm(trace)

        return edge, ampl, fwhm


    def measure_fwhm(self, trace: np.array) -> float:
        """! Calculate the full-width half-max of a convolution trace.

        @param trace (array-like) Convolution of image projection with Heaviside function.

        @return fwhm (float) Full-width half-max of signal in pixel space.
        """
        x = np.linspace(0, len(trace) - 1, len(trace))
        spline = UnivariateSpline(x, (trace - np.max(trace)/2), s=0)
        lb, rb = spline.roots()
        fwhm = rb - lb
        return fwhm

    def crop_image(self, img: np.array) -> np.array:
        """! Crop image. Currently done by inspection by specifying a range of
        rows containing the signal of interest.

        @param img (np.array) 2D time tool camera image.

        @return cropped (np.array) Cropped image.
        """
        colsum = np.sum(img, axis=1)
        argmaxi = colsum.argmax()

        s = np.max(((argmaxi - 15), 0))
        e = np.min(((argmaxi + 15), len(colsum) - 1))
        return img[s:e]

    def fit_calib(self, delays: np.array, edges: np.array, ampls: np.array,
                                          fwhm: np.array = None, order: int = 2,
                                                       xroi: list = [250, 870]):
        """! Fit a polynomial calibration curve to a list of edge pixel positions
            vs delay. In the future this will implement edge acceptance/rejection to
            improve the calibration. Currently only fits within a certain x pixel
            range of the camera.

        @param delays (list) List of TT target times (in ns) used for calibration.
        @param edges (list) List of corresponding detected edge positions on camera.
        @param ampls (list) Convolution amplitudes used for edge rejection.
        @param fwhms (list) Full-width half-max of convolution used for edge rejection.
        @param order (int) Order of polynomial to fit.
        @param xroi (list-like) X-window to consider for fitting. [Min X pixel, Max X pixel]
        """
        inrange = False

        self.edges = edges
        self.delays = delays
        if (edges > xroi[0]).any():
            delays_fit = delays[edges > xroi[0]]
            edges_fit = edges[edges > xroi[0]]
            if (edges_fit < xroi[1]).any():
                delays_fit = delays_fit[edges_fit < xroi[1]]
                edges_fit = edges_fit[edges_fit < xroi[1]]
                inrange = True

                self.edges_fit = edges_fit
                self.delays_fit = delays_fit
        if inrange:
            print('Fitting calibration between 250 and 870 pixels.')
            self._model = np.polyfit(edges_fit, delays_fit, order)
        else:
            print('Fit data is out of 250-870 pixel range. Results questionable.')
            self.edges_fit = self.edges
            self.delays_fit = self.delays
            self._model = np.polyfit(edges, delays, order)

    def ttstage_code(self, hutch: str) -> str:
        """! Return the correct code for the time tool delay (in ns) for a given
        experimental hutch.

        @param hutch (str) Three letter hutch name. E.g. mfx, cxi, xpp

        @return code (str) Epics code for accessing hutches time tool delay.
        """
        h = hutch.lower()
        num = ''

        if h == 'xpp':
            num = '3'
        elif h == 'xcs':
            num = '4'
        elif h == 'mfx':
            num = '45'
        elif h == 'cxi':
            num = '5'

        if num:
            return f'LAS:FS{num}:VIT:FS_TGT_TIME_DIAL'
        else:
            raise InvalidHutchError(hutch)

    def edge_to_time(self, edge_pos: int, dirtime: int = 1) -> float:
        """! Given an internal model and an edge position, return the time tool
        jitter correction.

        This method is called by the partner method actual_time_ps; however, it is
        provided directly so that users who have a different time convention
        (i.e. negative times when pump arrives before xray) can more easily
        correct their data.

        @param edge_pos (int | array-like) Position (pixel) of detected edge on camera.
        @param dirtime (int) Takes the value of 1 or -1. Use -1 to reverse the direction of time.

        @return correction (float) Jitter correction in picoseconds.
        """
        correction = 0
        if len(self._model) > 0:
            n = len(self._model)
            for i, coeff in enumerate(self._model):
                correction += coeff*edge_pos**(n-i-1)
            print('Using model to correct for jitter.')
        else:
            print('No calibration model. Jitter correction not possible.')

        correction *= 1000*dirtime
        return correction

    def actual_time_ps(self, edge_pos: int, nominal_delay: float) -> float:
        """! Return the actual time in picoseconds given a nominal delay and
        time tool data.

        @param edge_pos (int) Position (pixel) of detected edge on camera.
        @param nominal_delay (float) Nominal delay in picoseconds.

        @return time (float) Absolute time (ps). Nominal delay corrected for jitter.
        """
        time = nominal_delay
        time += self.edge_to_time(edge_pos)
        return time

    def timetool_correct(self, run: str, nominal: float, model = None,
                                                         figs: bool = True):
        """! Correct a run or set of runs at a given nominal delay for arrival
        time jitter. Outputs correct time stamps for later binning. Events are
        identified using their time and fiducial. The output text file has one
        event per line in the format (seconds-nanoseconds-fiducial tt_correction).

        @param run (str) Run(s) to correct with timetool. Single string, e.g. '17' or range '16-20'.
        @param nominal (float) Nominal time being corrected for in ps. E.g. .5 (500 fs)
        @param model (None | str | array-like) Polynomial coefficients of the timetool calibration model.
        @param figs (bool) Whether or not to produce diagnostic figures.
        """
        self.open_run(run)

        if model:
            self.model = model
        else:
            print('No model provided!')

        delays, edges, ampls, stamps = self.process_run(calib=False)
        times = self.actual_time_ps(edges, nominal)
        timed_stamps = np.array((stamps, times)).T

        run = self.format_run()
        fname = f'{run}.out'
        outdir = f'{self.savedir}/corrections'
        self.write_file(timed_stamps, fname, outdir, fmt='%s')
        if figs:
            self.plot_hist(edges)

    def plot_calib(self, delays: list, edges: list, model: list):
        """! Plot the density of detected edges during a time tool calibration
        run and the polynomial fit to it.

        @param delays (array-like) Time tool delays in nanoseconds.
        @param edges (array-like) Detected edges in time tool camera.
        @param model (array-like) Polynomial model coefficients.
        """
        poly = np.zeros([len(edges)])
        n = len(model)
        for i, coeff in enumerate(model):
            poly += coeff*edges**(n-i-1)

        fig, ax = plt.subplots(1, 1)
        ax.hexbin(edges, delays, gridsize=50, vmax=500, mincnt=1)
        ax.plot(edges, poly, 'ko', markersize=2)
        ax.set_xlabel('Edge Pixel')
        ax.set_ylabel('Delay (ns)')
        run = self.format_run()
        fname = f'TTCalib_{run}.png'
        self.write_file(fig, fname, f'{self.savedir}/figs')

    def plot_hist(self, edges: list):
        """! Plot the density of detected edges in a histogram. For calibration
        runs this should be uniform. For experiment runs this will ideally be
        quasi-Gaussian.

        @param edges (array-like) Detected edges in time tool camera.
        """
        fig, ax = plt.subplots(1, 1)
        ax.hist(edges, bins=20, density=True)
        ax.set_xlabel('Edge Pixel')
        ax.set_ylabel('Density')
        run = self.format_run()
        fname = f'EdgeHist_{run}.png'
        self.write_file(fig, fname, f'{self.savedir}/figs')

    def write_file(self, fobj, fname: str, savedir: str, fmt = None):
        """! Writes objects of multiple types to output files. Checks if the
        save directory exists, if not it creates it.

        @param fobj (plt.Figure | np.array) Object to write to disk. Can be a figure or array.
        @param fname (str) Output filename.
        @param savedir (str) Save directory.
        """
        try:
            os.makedirs(f'{savedir}')
        except FileExistsError:
            pass
        finally:
            if type(fobj) == plt.Figure:
                fobj.savefig(f'{savedir}/{fname}')
            elif type(fobj) == np.ndarray:
                if fmt:
                    np.savetxt(f'{savedir}/{fname}', fobj, fmt)
                else:
                    np.savetxt(f'{savedir}/{fname}', fobj)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val: list):
        """! Set the internal calibration model to coefficients which have been
        previously fit.

        @param val (list | str) List of polynomial coefficients in descending order, or file containing them.
        """
        if type(val) == list:
            # Directly setting via list
            self._model = model
        elif type(val) == str:
            # If stored in a file
            ext = val.split('.')[1]
            if ext == 'txt' or ext == 'out':
                self._model = np.loadtxt(val)
            elif ext == 'yaml':
                pass
            elif ext == 'npy':
                self._model = np.load(val)
        else:
            # Switch print statements to logging
            print("Entry not understood and model has not been changed.")



class InvalidHutchError(Exception):
    def __init__(self, hutch):
        pass

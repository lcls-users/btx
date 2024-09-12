import numpy as np
import logging
import psana
from psana import setOption
from psana import EventId
from PSCalib.GeometryAccess import GeometryAccess

logger = logging.getLogger(__name__)

class PsanaInterface:

    def __init__(self, exp, run, det_type,
                 event_receiver=None, event_code=None, event_logic=True,
                 ffb_mode=False, track_timestamps=False, hutch=None,
                 xtcdir=None, calibdir=None):
        self.exp = exp # experiment name, str
        if hutch is None:
            self.hutch = exp[:3] # hutch name, str
        else:
            self.hutch = hutch
        self.run = run # run number, int
        self.det_type = det_type # detector name, str
        self.track_timestamps = track_timestamps # bool, keep event info
        self.seconds, self.nanoseconds, self.fiducials = [], [], []
        self.event_receiver = event_receiver # 'evr0' or 'evr1', str
        self.event_code = event_code # event code, int
        self.event_logic = event_logic # bool, if True, retain events with event_code; if False, keep all other events
        self.set_up(det_type, xtcdir, calibdir)
        self.counter = 0

    def set_up(self, det_type, xtcdir=None, calibdir=None):
        """
        Instantiate DataSource and Detector objects; use the run 
        functionality to retrieve all psana.EventTimes.
        
        Parameters
        ----------
        det_type : str
            detector type, e.g. epix10k2M or jungfrau4M
        xtcdir: str
            directory to alternative XTC files
        calibdir: str
            directory to alternative calibration files
        """
        ds_args=f'exp={self.exp}:run={self.run}:idx'
        if xtcdir is not None:
            ds_args += f':dir={xtcdir}'
        #if ffb_mode:
        #    ds_args += f':dir=/cds/data/drpsrcf/{self.exp[:3]}/{self.exp}/xtc'
        
        self.ds = psana.DataSource(ds_args)   
        self.det = psana.Detector(det_type, self.ds.env())
        if self.event_receiver is not None:
            self.evr_det = psana.Detector(self.event_receiver)
        self.runner = next(self.ds.runs())
        self.times = self.runner.times()
        self.max_events = len(self.times)
        if calibdir is not None:
            setOption('psana.calib_dir', calibdir)
        self._calib_data_available()

    def _calib_data_available(self):
        """
        Check whether calibration data is available.
        """
        self.calibrate = True
        evt = self.runner.event(self.times[0])
        if (self.det.pedestals(evt) is None) or (self.det.gain(evt) is None):
            logger.warning("Warning: calibration data unavailable, returning uncalibrated data")
            self.calibrate = False

    def turn_calibration_off(self):
        """
        Do not apply calibration to images.
        """
        self.calibrate = False

    def get_pixel_size(self):
        """
        Retrieve the detector's pixel size in millimeters.

        Returns
        -------
        pixel_size : float
            detector pixel size in mm
        """
        if self.det_type.lower() == 'rayonix':
            env = self.ds.env()
            cfg = env.configStore()
            pixel_size_um = cfg.get(psana.Rayonix.ConfigV2).pixelWidth()
        else:
            pixel_size_um = self.det.pixel_size(self.ds.env())
        return pixel_size_um / 1.0e3

    
    def get_wavelength(self):
        """
        Retrieve the detector's wavelength in Angstrom.

        Returns
        -------
        wavelength : float
            wavelength in Angstrom
        """
        return self.ds.env().epicsStore().value('SIOC:SYS0:ML00:AO192') * 10.
    
    def get_wavelength_evt(self, evt):
        """
        Retrieve the detector's wavelength for a specific event.

        Parameters
        ----------
        evt : psana.Event object
            individual psana event

        Returns
        -------
        wavelength : float
            wavelength in Angstrom
        """
        photon_energy = self.get_photon_energy_eV_evt(evt)
        if photon_energy is None or np.isinf(photon_energy):
            return self.get_wavelength()
        else:
            lambda_m =  1.23984197386209e-06 / photon_energy # convert to meters using e=hc/lambda
            return lambda_m * 1e10

    def get_photon_energy_eV_evt(self, evt):
        """
        Retrieve the photon energy in eV for a specific event.
        Parameters
        ----------
        evt : psana.Event object
            individual psana event

        Returns
        -------
        photon_energy : float
            photon energy in eV
        """
        try:
            return psana.Detector('EBeam').get(evt).ebeamPhotonEnergy()
        except AttributeError as e:
            logger.warning("Event lacking an ebeamPhotonEnergy value.")

    def get_fee_gas_detector_energy_mJ_evt(self, evt, mode=None):
        """
        Retrieve pulse energy measured by Front End Enclosure Gas Detectors.
        For more information:
         - https://pswww.slac.stanford.edu/swdoc/releases/ana-current/psana-ref/html/psana/#class-psana-bld-blddatafeegasdetenergyv1
         - https://confluence.slac.stanford.edu/display/PSDM/New+XTCAV+Documentation
         - https://www-ssrl.slac.stanford.edu/lcls/technotes/LCLS-TN-09-5.pdf

        Note: the pulse energy is equal to the photon energy times the number of photons in the pulse.

        Parameters
        ----------
        evt : psana.Event object
            individual psana event
        mode : str, optional
            whether to only return the energy 'before' or 'after' gas attenuation,
            or if 'None' the average of the two.

        Returns
        -------
        gas_detector_energy: float
            Beam energy in mJ

        """
        gdet = evt.get(psana.Bld.BldDataFEEGasDetEnergyV1, psana.Source())
        if gdet is not None:
            gdet_before_attenuation = 0.5 * (gdet.f_11_ENRC() + gdet.f_12_ENRC())
            gdet_after_attenuation  = 0.5 * (gdet.f_21_ENRC() + gdet.f_22_ENRC())
            if( mode == 'before' ):
                return gdet_before_attenuation
            elif( mode == 'after' ):
                return gdet_after_attenuation
            else:
                return 0.5 * (gdet_before_attenuation + gdet_after_attenuation)

    def estimate_distance(self):
        """
        Retrieve an estimate of the detector distance in mm.

        Returns
        -------
        distance : float
            estimated detector distance
        """
        return -1*np.mean(self.det.coords_z(self.run))/1e3

    def get_camera_length(self, pv_camera_length=None):

        """
        Retrieve the camera length (clen) in mm.

        Parameters
        ----------
        pv_camera_length : str
            PV associated with camera length

        Returns
        -------
        clen : float
            clen, where clen = distance - coffset in mm
        """
        if pv_camera_length is None:
            if self.det_type == 'jungfrau4M':
                pv_camera_length = 'CXI:DS1:MMS:06.RBV'
            if self.det_type == 'Rayonix':
                pv_camera_length = 'MFX:DET:MMS:04.RBV'
            if self.det_type == 'epix10k2M':
                pv_camera_length = 'MFX:ROB:CONT:POS:Z'
            logger.debug(f"PV used to retrieve clen parameter: {pv_camera_length}")

        try:
            return self.ds.env().epicsStore().value(pv_camera_length)
        except TypeError:
            raise RuntimeError(f"PV {pv_camera_length} is invalid")

    def get_beam_transmission(self, pv_beam_transmission=None):
        """
        Fraction of beam transmitted to the sample.
        The attenuation is set by beamline scientists before the beam reaches the sample.

        Parameters
        ----------
        pv_beam_transmission : str
            PV associated with beam attenuation

        Returns
        -------
        beam_transmission : float
            0.0 is no beam, 1.0 is full beam.
        """
        if pv_beam_transmission is None:
            if self.hutch == 'mfx':
                pv_beam_transmission = "MFX:ATT:COM:R_CUR"
            elif self.hutch == 'cxi':
                pv_beam_transmission = "CXI:DIA:ATT:COM:R_CUR"
            else:
                raise NotImplementedError

        try:
            return self.ds.env().epicsStore().value(pv_beam_transmission)
        except TypeError:
            raise RuntimeError(f"PV {pv_beam_transmission} is invalid")

    def get_timestamp(self, evtId):
        """
        Retrieve the timestamp (seconds, nanoseconds, fiducials) associated with the input 
        event and store in self variables. For further details, see the example here:
        https://confluence.slac.stanford.edu/display/PSDM/Jump+Quickly+to+Events+Using+Timestamps
        
        Parameters
        ----------
        evtId : psana.EventId
            the event ID associated with a particular image
        """
        self.seconds.append(evtId.time()[0])
        self.nanoseconds.append(evtId.time()[1])
        self.fiducials.append(evtId.fiducials())
        return

    def skip_event(self, evt):
        """
        We skip an event if:
        - it has a specific event_code and our event_logic is False
        - it does not have that event_code and our event_logic is True.

        Parameters
        ----------
        evt : psana.Event object
            individual psana event

        Returns
        -------
        skip_status : Boolean
            if True, skip event
        """
        skip = False
        if self.event_receiver is not None:
            event_codes = self.evr_det.eventCodes(evt)
            found_event = False
            if self.event_code in event_codes:
                found_event = True
            if ( found_event != self.event_logic ):
                skip = True
        return skip

    def distribute_events(self, rank, total_ranks, max_events=-1):
        """
        For parallel processing. Update self.counter and self.max_events such that
        events will be distributed evenly across total_ranks, and each rank will 
        only process its assigned events. Hack to avoid explicitly using MPI here.
        
        Parameters
        ----------
        rank : int
            current rank
        total_ranks : int
            total number of ranks
        max_events : int, optional
            total number of images desired, option to override self.max_events 
        """
        if max_events == -1:
            max_events = self.max_events
            
        # determine boundary indices between ranks
        split_indices = np.zeros(total_ranks)
        for r in range(total_ranks):
            num_per_rank = max_events // total_ranks
            if r < (max_events % total_ranks):
                num_per_rank += 1
            split_indices[r] = num_per_rank

        split_indices = np.append(np.array([0]), np.cumsum(split_indices)).astype(int)   
        
        # update self variables that determine start and end of this rank's batch
        self.counter = split_indices[rank]
        self.max_events = split_indices[rank+1]
        
    def get_images(self, num_images, assemble=True):
        """
        Retrieve a fixed number of images from the run. If the pedestal or gain 
        information is unavailable and unassembled images are requested, return
        uncalibrated images. 

        Parameters
        ---------
        num_images : int
            number of images to retrieve (per rank)
        assemble : bool, default=True
            whether to assemble panels into image

        Returns
        -------
        images : numpy.ndarray, shape ((num_images,) + det_shape)
            images retrieved sequentially from run, optionally assembled
        """
        # set up storage array
        if 'opal' not in self.det_type.lower():
            if assemble:
                images = np.zeros((num_images, 
                                   self.det.image_xaxis(self.run).shape[0], 
                                   self.det.image_yaxis(self.run).shape[0]))
            else:
                images = np.zeros((num_images,) + self.det.shape())
        else:
            images = np.zeros((num_images, 230, 1024))
            assemble = False

        # retrieve next batch of images
        counter_batch = 0
        while counter_batch < num_images:

            if self.counter >= self.max_events:
                images = images[:counter_batch]
                print("No more events to retrieve")
                break
            else:
                evt = self.runner.event(self.times[self.counter])
                if assemble and self.det_type.lower()!='rayonix':
                    if not self.calibrate:
                        raise IOError("Error: calibration data not found for this run.")
                    else:
                        img = self.det.image(evt=evt)
                else:
                    if self.calibrate:
                        img = self.det.calib(evt=evt)
                    else:
                        img = self.det.raw(evt=evt)
                        if self.det_type == 'epix10k2M':
                            img = img & 0x3fff # exclude first two bits
                if img is not None:
                    images[counter_batch] = img
                    counter_batch += 1
                        
                if self.track_timestamps:
                    self.get_timestamp(evt.get(EventId))
                    
                self.counter += 1
             
        return images

#### Miscellaneous functions ####

def retrieve_pixel_index_map(geom):
    """
    Retrieve a pixel index map that specifies the relative arrangement of
    pixels on an LCLS detector.

    Parameters
    ----------
    geom : string or GeometryAccess Object
        if str, full path to a psana *-end.data file
        else, a PSCalib.GeometryAccess.GeometryAccess object

    Returns
    -------
    pixel_index_map : numpy.ndarray, 4d or 5d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)
                           shape (pidx1, pidx2, fs_shape, ss_shape, 2)
    """
    if type(geom) == str:
        geom = GeometryAccess(geom)

    temp_index = [np.asarray(t) for t in geom.get_pixel_coord_indexes()]
    pixel_index_map = np.zeros((np.array(temp_index).shape[2:]) + (2,))
    pixel_index_map[...,0] = temp_index[0][0]
    pixel_index_map[...,1] = temp_index[1][0]
    
    return pixel_index_map.astype(np.int64)


def assemble_image_stack_batch(image_stack, pixel_index_map):
    """
    Assemble the image stack to obtain a 2D pattern according to the index map.
    Either a batch or a single image can be provided. Modified from skopi.

    Parameters
    ----------
    image_stack : numpy.ndarray, 3d or 4d
        stack of images, shape (n_images, n_panels, fs_panel_shape, ss_panel_shape)
        or (n_panels, fs_panel_shape, ss_panel_shape)
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)

    Returns
    -------
    images : numpy.ndarray, 3d
        stack of assembled images, shape (n_images, fs_panel_shape, ss_panel_shape)
        of shape (fs_panel_shape, ss_panel_shape) if ony one image provided
    """
    multiple_panel_dimensions = False
    if len(image_stack.shape) == 3:
        image_stack = np.expand_dims(image_stack, 0)

    if len(pixel_index_map.shape) == 5:
        multiple_panel_dimensions = True
        
    # get boundary
    index_max_x = np.max(pixel_index_map[..., 0]) + 1
    index_max_y = np.max(pixel_index_map[..., 1]) + 1
    # get stack number and panel number
    stack_num = image_stack.shape[0]

    # set holder
    images = np.zeros((stack_num, index_max_x, index_max_y))

    if multiple_panel_dimensions:
        pdim1 = pixel_index_map.shape[0]
        pdim2 = pixel_index_map.shape[1]
        for i in range(pdim1):
            for j in range(pdim2):
                x = pixel_index_map[i, j, ..., 0]
                y = pixel_index_map[i, j, ..., 1]
                idx = i*pdim2 + j
                images[:, x, y] = image_stack[:, idx]
    else:
        panel_num = image_stack.shape[1]
        # loop through the panels
        for l in range(panel_num):
            x = pixel_index_map[l, ..., 0]
            y = pixel_index_map[l, ..., 1]
            images[:, x, y] = image_stack[:, l]

    if images.shape[0] == 1:
        images = images[0]

    return images

def disassemble_image_stack_batch(images, pixel_index_map):
    """
    Diassemble a series of 2D diffraction patterns into their consituent panels. 
    Function modified from skopi.

    Parameters
    ----------
    images : numpy.ndarray, 3d
        stack of assembled images, shape (n_images, fs_panel_shape, ss_panel_shape)
        of shape (fs_panel_shape, ss_panel_shape) if ony one image provided
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)

    Returns
    -------
    image_stack_batch : numpy.ndarray, 3d or 4d 

        stack of images, shape (n_images, n_panels, fs_panel_shape, ss_panel_shape)
        or (n_panels, fs_panel_shape, ss_panel_shape)
    """
    multiple_panel_dimensions = False
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)

    if len(pixel_index_map.shape) == 5:
        multiple_panel_dimensions = True

    if multiple_panel_dimensions:
        ishape = images.shape[0]
        (pdim1, pdim2, fs_shape, ss_shape) = pixel_index_map.shape[:-1]
        image_stack_batch = np.zeros((ishape, pdim1*pdim2, fs_shape, ss_shape))
        for i in range(pdim1):
            for j in range(pdim2):
                x = pixel_index_map[i, j, ..., 0]
                y = pixel_index_map[i, j, ..., 1]
                idx = i*pdim2 + j
                image_stack_batch[:, idx] = images[:, x, y]
    else:
        image_stack_batch = np.zeros((images.shape[0],) + pixel_index_map.shape[:3])
        for panel in range(pixel_index_map.shape[0]):
            idx_map_1 = pixel_index_map[panel, :, :, 0]
            idx_map_2 = pixel_index_map[panel, :, :, 1]
            image_stack_batch[:, panel] = images[:, idx_map_1, idx_map_2]

    if image_stack_batch.shape[0] == 1:
        image_stack_batch = image_stack_batch[0]

    return image_stack_batch

#### binning methods ####

def bin_data(arr, bin_factor, det_shape=None):
    """
    Bin detector data by bin_factor through averaging.
    Retrieved from
    https://github.com/apeck12/cmtip/blob/main/cmtip/prep_data.py

    :param arr: array shape (n_images, n_panels, panel_shape_x, panel_shape_y)
      or if det_shape is given of shape (n_images, 1, n_pixels_per_image)
    :param bin_factor: how may fold to bin arr by
    :param det_shape: tuple of detector shape, optional
    :return arr_binned: binned data of same dimensions as arr
    """
    # reshape as needed
    if det_shape is not None:
        arr = np.array([arr[i].reshape(det_shape) for i in range(arr.shape[0])])

    n, p, y, x = arr.shape

    # ensure that original shape is divisible by bin factor
    assert y % bin_factor == 0
    assert x % bin_factor == 0

    # bin each panel of each image
    binned_arr = (
        arr.reshape(
            n,
            p,
            int(y / bin_factor),
            bin_factor,
            int(x / bin_factor),
            bin_factor,
        )
        .mean(-1)
        .mean(3)
    )

    # if input data were flattened, reflatten
    if det_shape is not None:
        flattened_size = np.prod(np.array(binned_arr.shape[1:]))
        binned_arr = binned_arr.reshape((binned_arr.shape[0], 1) + (flattened_size,))

    return binned_arr


def bin_pixel_index_map(arr, bin_factor):
    """
    Bin pixel_index_map by bin factor.
    Retrieved from
    https://github.com/apeck12/cmtip/blob/main/cmtip/prep_data.py

    :param arr: pixel_index_map of shape (n_panels, panel_shape_x, panel_shape_y, 2)
    :param bin_factor: how may fold to bin arr by
    :return binned_arr: binned pixel_index_map of same dimensions as arr
    """
    arr = np.moveaxis(arr, -1, 0)
    if bin_factor > 1:
        arr = np.minimum(arr[..., ::bin_factor, :], arr[..., 1::bin_factor, :])
        arr = np.minimum(arr[..., ::bin_factor], arr[..., 1::bin_factor])
        arr = arr // bin_factor

    return np.moveaxis(arr, 0, -1)

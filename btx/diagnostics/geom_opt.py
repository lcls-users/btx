import numpy as np
import sys
import requests
from mpi4py import MPI
from btx.diagnostics.run import RunDiagnostics
from btx.interfaces.ipsana import assemble_image_stack_batch
from btx.misc.metrology import *
from btx.misc.radial import pix2q
from .ag_behenate import *
import itertools

class GeomOpt:
    
    def __init__(self, exp, run, det_type):
        self.diagnostics = RunDiagnostics(exp=exp, # experiment name, str
                                          run=run, # run number, int
                                          det_type=det_type) # detector name, str
        self.center = None
        self.distance = None
        self.edge_resolution = None

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
    def distribute_scan(self, scan):
        """
        Distribute the scan across all ranks.
        
        Parameters
        ----------
        scan : list of tuples
            parameters (n_peak, center, distance) for initial estimates
        """
        n_search = len(scan)
        split_indices = np.zeros(self.size) 
        for r in range(self.size):
            num_per_rank = n_search // self.size
            if r < (n_search % self.size):
                num_per_rank += 1
            split_indices[r] = num_per_rank
        split_indices = np.append(np.array([0]), np.cumsum(split_indices)).astype(int) 
        return scan[split_indices[self.rank]:split_indices[self.rank+1]]
        
    def opt_geom(self, powder, mask=None, center=None, distance=None, n_iterations=5, 
                 n_peaks=[3], threshold=1e6, deltas=True, plot=None, plot_final_only=False):
        """
        Estimate the sample-detector distance based on the properties of the powder
        diffraction image. Currently only implemented for silver behenate.
        
        Parameters
        ----------
        powder : str or int
            if str, path to the powder diffraction in .npy format
            if int, number of images from which to compute powder 
        mask : str
            npy file of mask in psana unassembled detector shape
        center : tuple
            list of detector center(s) (xc,yc) in pixels as starting guess 
            if None, assume assembled image center.
        distance : float
            list of the intial estimate(s) of the sample-detector distance 
            in mm. If None, pull from calib file.
        n_iterations : int
            number of refinement steps
        n_peaks : int
            list of the number of observed peaks to use for center fitting
        threshold : float
            pixels above this intensity in powder get set to 0; None for no thresholding.
        deltas: bool
            whether centers are in absolute positions or delta pixels from detector center
        plot : str or None
            output path for figure; if '', plot but don't save; if None, don't plot
        plot_final_only: bool
            if True, only generate plot for the best distance / center
        """
        
        if type(powder) == str:
            if os.path.isfile(powder):
                print("Loading powder from {powder}")
                powder_img = np.load(powder)
            else:
                print("Computing powder over all events")
                self.diagnostics.compute_run_stats(powder_only=True)
                if self.diagnostics.psi.det_type != 'Rayonix':
                    powder_img = assemble_image_stack_batch(self.diagnostics.powders['max'],
                                                            self.diagnostics.pixel_index_map)
        elif type(powder) == int:
            print("Computing powder over {powder} events")
            self.diagnostics.compute_run_stats(max_events=powder, powder_only=True)
            if self.diagnostics.psi.det_type != 'Rayonix':
                powder_img = assemble_image_stack_batch(self.diagnostics.powders['max'], 
                                                        self.diagnostics.pixel_index_map)
        else:
            sys.exit("Unrecognized powder type, expected a path or number")
        self.img_shape = powder_img.shape
        
        if mask:
            print(f"Loading mask {mask}")
            mask = np.load(mask)
            if self.diagnostics.psi.det_type != 'Rayonix':
                mask = assemble_image_stack_batch(mask, self.diagnostics.pixel_index_map)

        if distance is None:
            distance = [self.diagnostics.psi.estimate_distance()]

        midpt = np.array(powder_img.shape)/2
        if center is None:
            center = tuple(midpt)
        else:
            if deltas:
                center = [(c[0]+midpt[0], c[1]+midpt[1]) for c in center]

        scan = list(itertools.product(n_peaks, center, distance))
        scan_rank = self.distribute_scan(scan)

        plot_intermediate=plot
        if plot_final_only:
            plot_intermediate=None
        
        ag_behenate = AgBehenate(powder_img,
                                 mask,
                                 self.diagnostics.psi.get_pixel_size(),
                                 self.diagnostics.psi.get_wavelength())
        
        if len(scan_rank) > 0:
            for params in scan_rank:
                ag_behenate.opt_geom(params[2], 
                                     n_iterations=n_iterations, 
                                     n_peaks=params[0], 
                                     threshold=threshold, 
                                     center_i=params[1],
                                     plot=plot_intermediate)
        self.comm.Barrier()
        
        self.scan = {}
        self.scan['distance'] = self.comm.gather(ag_behenate.distances, root=0) # in mm
        self.scan['center'] = self.comm.gather(ag_behenate.centers, root=0) # in pixels
        self.scan['npeaks'] = self.comm.gather(ag_behenate.npeaks, root=0)
        self.scan['scores_min'] = self.comm.gather(ag_behenate.scores_min, root=0)
        self.scan['scores_mean'] = self.comm.gather(ag_behenate.scores_mean, root=0)
        self.scan['scores_std'] = self.comm.gather(ag_behenate.scores_std, root=0)

        self.finalize()
        if self.rank == 0:
            # generate plot based on best geometry
            ag_behenate.centers.append(self.center)
            ag_behenate.distances.append(self.distance)
            ag_behenate.opt_distance(plot=plot)

    def finalize(self):
        """
        Compute the final score based on how many standard deviations the 
        minimum is from the mean in the inter-ring spacing analysis. Also
        store the optimal distance, center, and edge resolution.
        """
        if self.rank == 0:
            for key in self.scan.keys():
                self.scan[key] = np.array([item for sublist in self.scan[key] for item in sublist])
            invalid = np.isnan(self.scan['scores_min'])
            for key in self.scan.keys():
                self.scan[key] = self.scan[key][~invalid]
            self.scan['scores_final'] = (self.scan['scores_mean'] - self.scan['scores_min']) / self.scan['scores_std']
            
            index = np.argmax(self.scan['scores_final']) 
            self.distance = self.scan['distance'][index] # in mm
            self.center = self.scan['center'][index] # in pixels
            self.edge_resolution = 1.0 / pix2q(np.array([self.img_shape[0]/2]),
                                               self.diagnostics.psi.get_wavelength(), 
                                               self.distance,
                                               self.diagnostics.psi.get_pixel_size())[0] # in Angstrom
            
    def deploy_geometry(self, outdir, pv_camera_length=None):
        """
        Write new geometry files (.geom and .data for CrystFEL and psana respectively) 
        with the optimized center and distance.
    
        Parameters
        ----------
        outdir : str
            path to output directory
        pv_camera_length : str
            PV associated with camera length
        """
        # retrieve original geometry
        run = self.diagnostics.psi.run
        geom = self.diagnostics.psi.det.geometry(run)
        top = geom.get_top_geo()
        children = top.get_list_of_children()[0]
        pixel_size = self.diagnostics.psi.get_pixel_size() * 1e3 # from mm to microns
    
        # determine and deploy shifts in x,y,z
        cy, cx = self.diagnostics.psi.det.point_indexes(run, pxy_um=(0,0), fract=True)
        dx = pixel_size * (self.center[0] - cx) # convert from pixels to microns
        dy = pixel_size * (self.center[1] - cy) # convert from pixels to microns
        dz = np.mean(-1*self.diagnostics.psi.det.coords_z(run)) - 1e3 * self.distance # convert from mm to microns
        geom.move_geo(children.oname, 0, dx=-dy, dy=-dx, dz=dz) 
    
        # write optimized geometry files
        psana_file, crystfel_file = os.path.join(outdir, f"r{run:04}_end.data"), os.path.join(outdir, f"r{run:04}.geom")
        temp_file = os.path.join(outdir, "temp.geom")
        geom.save_pars_in_file(psana_file)
        generate_geom_file(self.diagnostics.psi.exp, run, self.diagnostics.psi.det_type, psana_file, temp_file, pv_camera_length)
        modify_crystfel_header(temp_file, crystfel_file)
        os.remove(temp_file)

        # Rayonix check
        if self.diagnostics.psi.get_pixel_size() != self.diagnostics.psi.det.pixel_size(run):
            print("Original geometry is wrong due to hardcoded Rayonix pixel size. Correcting geom file now...")
            coffset = (self.distance - self.diagnostics.psi.get_camera_length(pv_camera_length)) / 1e3 # convert from mm to m
            res = 1e3 / self.diagnostics.psi.get_pixel_size() # convert from mm to um
            os.rename(crystfel_file, temp_file)
            modify_crystfel_coffset_res(temp_file, crystfel_file, coffset, res)
            os.remove(psana_file)
            os.remove(temp_file)

    def report(self, update_url):
        """
        Post summary to elog.
       
        Parameters
        ----------
        update_url : str
            elog URL for posting progress update
        """
        requests.post(update_url, json=[{ "key": "Detector distance (mm)", "value": f"{self.distance:.2f}" },
                                        { "key": "Detector center (pixels)", "value": f"({self.center[0]:.2f}, {self.center[1]:.2f})" },
                                        { "key": "Detector edge resolution (A)", "value": f"{self.edge_resolution:.2f}" }, ])

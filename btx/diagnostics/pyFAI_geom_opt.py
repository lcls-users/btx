import sys
import numpy as np
import matplotlib.pyplot as plt
import pyFAI
from pyFAI.calibrant import CalibrantFactory, CALIBRANT_FACTORY
from pyFAI.goniometer import SingleGeometry
from pyFAI.geometry import Geometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.gui import jupyter
from btx.interfaces.ipsana import *
from btx.diagnostics.run import RunDiagnostics
from btx.diagnostics.converter import CrystFELtoPyFAI
from btx.misc.metrology import *


class PyFAIGeomOpt:
    """
    Class to perform Geometry Optimization using pyFAI

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    detector : str
        Detector type
    geom : str
        Geometry file in CrystFEL format
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
    ):
        self.diagnostics = RunDiagnostics(exp, run, det_type=det_type)
        self.distance = None
        self.poni1 = None
        self.poni2 = None

    def pyFAI_geom(self, geom):
        """
        Load geometry from CrystFEL format
        Convert to pyFAI format

        Parameters
        ----------
        geom : str
            Geometry file in CrystFEL format
        """
        converter = CrystFELtoPyFAI(geom)
        detector = converter.detector
        detector.set_pixel_corners(converter.corner_array)
        self.detector = detector
        return detector

    def point_index_to_coords(self, point=(0, 0)):
        """
        Convert pixel index (x, y) to cartesian coordinates (x, y, z)

        Parameters
        ----------
        point : tuple
            Pixel index (x, y)
        """
        return self.detector.calc_cartesian_positions(d1=point[1], d2=point[0], center=True)

    def pyFAI_geom_opt(self, powder, mask=None, max_rings=5, pts_per_deg=1, I=0, plot=None):
        """
        From guessed initial geometry, optimize the geometry using pyFAI package

        Parameters
        ----------
        max_rings : int
            Maximum number of rings to use for calibration
        pts_per_deg : int
            Number of points per degree to use for calibration (spacing of control points)
        Imin : float
            Minimum intensity to use for calibration
        """
        # 1. Define Calibrant
        behenate = CALIBRANT_FACTORY("AgBh")
        wavelength = self.diagnostics.psi.get_wavelength() * 1e-10
        photon_energy = 1.23984197386209e-09 / wavelength
        behenate.wavelength = wavelength

        # 2. Define Guessed Geometry
        p1, p2, p3 = self.detector.calc_cartesian_positions()
        distance = np.mean(p3)
        poni1 = np.mean(p1)
        poni2 = np.mean(p2)
        guessed_geom = Geometry(
            dist=distance,
            poni1=poni1,
            poni2=poni2,
            detector=self.detector,
            wavelength=wavelength,
        )

        # 3. Powder Loading
        if type(powder) == str:
            powder_img = np.load(powder)
        elif type(powder) == int:
            print("Computing powder from scratch")
            self.diagnostics.psi.calibrate = True
            powder_imgs = self.diagnostics.psi.get_images(powder, assemble=False)
            powder_img = np.max(powder_imgs, axis=0)
            powder_img = np.reshape(powder_img, self.detector.shape)
        else:
            sys.exit("Unrecognized powder type, expected a path or number")
        self.img_shape = powder_img.shape

        if mask:
            print(f"Loading mask {mask}")
            mask = np.load(mask)
        else:
            mask = self.diagnostics.psi.get_mask()
        powder_img *= mask

        # 3. PyFAI Optimization
        pixel_size = min(self.detector.pixel1, self.detector.pixel2)
        print(
            f"Starting optimization with initial guess: dist={self.distance:.3f}m, poni1={self.poni1/pixel_size:.3f}pix, poni2={self.poni2/pixel_size:.3f}pix"
        )
        sg = SingleGeometry(
            label="AgBh",
            image=powder_img,
            calibrant=behenate,
            detector=self.detector,
            geometry=guessed_geom,
        )
        sg.extract_cp(
            max_rings=max_rings, pts_per_deg=pts_per_deg, Imin=I * photon_energy
        )
        score = sg.geometry_refinement.refine3(
            fix=["rot1", "rot2", "rot3", "wavelength"]
        )
        print(
            f"Optimization complete with final parameters: dist={distance:.3f}m, poni1={poni1/pixel_size:.3f}pix, poni2={poni2/pixel_size:.3f}pix"
        )
        self.distance = sg.geometry_refinement.param[0]
        self.poni1 = sg.geometry_refinement.param[1]
        self.poni2 = sg.geometry_refinement.param[2]

        # 4. Plotting
        if plot is not None:
            fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            ax[0].jupyter.display(sg=sg)
            ai = AzimuthalIntegrator(dist=self.distance, poni1=self.poni1, poni2=self.poni2, detector=self.detector, wavelength=wavelength)
            res = ai.integrate1d(powder_img, 1000, unit="2th_deg", calibrant=behenate)
            ax[1].jupyter.plot1d(res)
            fig.savefig(plot)
        return score

    def deploy_geometry(self, geom_init, outdir, pv_camera_length=None):
        """
        Write new geometry files (.geom and .data for CrystFEL and psana respectively)
        with the optimized center and distance.

        Parameters
        ----------
        geom_init : str
            Initial geometry file in CrystFEL format
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
        pixel_size = self.diagnostics.psi.get_pixel_size() * 1e3  # from mm to microns

        # determine and deploy shifts in x,y,z
        p1, p2, p3 = self.detector.calc_cartesian_positions()
        dx = (self.poni2 - np.mean(p2)) * 1e6
        dy = (self.poni1 - np.mean(p1)) * 1e6
        dz = self.distance * 1e6 - np.mean(p3)  # convert from m to microns
        geom.move_geo(
            children.oname, 0, dx=-dy, dy=-dx, dz=-dz
        )  # move the detector in psana frame

        # write optimized geometry files
        psana_file, crystfel_file = os.path.join(
            outdir, f"r{run:04}_end.data"
        ), os.path.join(outdir, f"r{run:04}.geom")
        temp_file = os.path.join(outdir, "temp.geom")
        geom.save_pars_in_file(psana_file)
        generate_geom_file(
            self.diagnostics.psi.exp,
            run,
            self.diagnostics.psi.det_type,
            psana_file,
            temp_file,
            pv_camera_length,
        )
        modify_crystfel_header(temp_file, crystfel_file)
        os.remove(temp_file)

        # Rayonix check
        if self.diagnostics.psi.get_pixel_size() != self.diagnostics.psi.det.pixel_size(
            run
        ):
            print(
                "Original geometry is wrong due to hardcoded Rayonix pixel size. Correcting geom file now..."
            )
            coffset = (
                self.distance - self.diagnostics.psi.get_camera_length(pv_camera_length)
            ) / 1e3  # convert from mm to m
            res = 1e3 / self.diagnostics.psi.get_pixel_size()  # convert from mm to um
            os.rename(crystfel_file, temp_file)
            modify_crystfel_coffset_res(temp_file, crystfel_file, coffset, res)
            os.remove(psana_file)
            os.remove(temp_file)

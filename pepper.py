import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

def emittance(x, xp):
    return np.sqrt(np.linalg.det(np.cov(x,xp)))

class Pepperpot:
    """
    Simplified simulation of a pepperpot
    """

    def __init__(self, d=2, r=0.1, l=190, n=31):
        """
        Pepperpot parameters
        d = distance betweens holes (square grid)
        r = hole radius
        l = distance mask screen
        n = number of holes in each direction
        all lengths in mm
        """
        self._r = r
        self._l = l
        self._d = d

        n = n - 1
        cen = np.arange(-n/2*d, (n/2 + 1)*d, d)
        # cen_x, cen_y = np.meshgrid(cen, cen)
        self._cen = {}
        self._cen["x"] = cen# np.ravel(cen_x)
        self._cen["y"] = cen# np.ravel(cen_y)

    def find_closest_centres(self, p):
        """
        finds the closest centre positions for each particle
        """
        inds_x = np.digitize(p["x"], self._cen["x"] + self._d/2)
        inds_y = np.digitize(p["y"], self._cen["y"] + self._d/2)
        x_cen = self._cen["x"][inds_x]
        y_cen = self._cen["y"][inds_y]
        return {"x":x_cen, "y":y_cen}

    def mask_particles(self, p):
        """
        Selects and returns the particles that pass the mask
        p = dict type particle data
        """
        cen = self.find_closest_centres(p)
        dist = np.sqrt((p["x"] - cen["x"])**2 + (p["y"] - cen["y"])**2)
        msk = dist <= self._r
        p_out = {}
        p_out["x"] = np.extract(msk, p["x"])
        p_out["y"] = np.extract(msk, p["y"])
        p_out["xp"] = np.extract(msk, p["xp"])
        p_out["yp"] = np.extract(msk, p["yp"])
        return p_out

    def project_on_screen(self, p):
        """
        Projects coordinates from the mask into the screen plane
        p = dict type particle data
        """
        p_out = {}
        p_out["x"] = p["x"] + self._l*np.tan(p["xp"])
        p_out["y"] = p["y"] + self._l*np.tan(p["yp"])
        p_out["xp"] = p["xp"][:]
        p_out["yp"] = p["yp"][:]
        return p_out

    def reconstruct_phase_space(self, p):
        """
        Reconstructs the phase space coordinates in the mask plane given the screen plane coord.
        p = dict type particle data
        """
        p_out = {}
        cen = self.find_closest_centres(p)
        p_out["x"] = cen["x"]
        p_out["y"] = cen["y"]
        p_out["xp"] = np.arctan2(p["x"]-cen["x"], self._l)
        p_out["yp"] = np.arctan2(p["y"]-cen["y"], self._l)
        return p_out

    def measurement(self, p_inp):
        p_msk = self.mask_particles(p_inp)
        p_scr = self.project_on_screen(p_msk)
        p_rec = self.reconstruct_phase_space(p_scr)
        emit_inp_x = 1000 * emittance(p_inp["x"], p_inp["xp"])
        emit_inp_y = 1000 * emittance(p_inp["y"], p_inp["yp"])
        emit_rec_x = 1000 * emittance(p_rec["x"], p_rec["xp"])
        emit_rec_y = 1000 * emittance(p_rec["y"], p_rec["yp"])
        res = {"p_inp":p_inp, "p_msk":p_msk, "p_scr":p_scr, "p_rec": p_rec,
               "emit_inp_x": emit_inp_x, "emit_inp_y": emit_inp_y,
               "emit_rec_x": emit_rec_x, "emit_rec_y": emit_rec_y}
        return res

    def show_mask(self):
        """
        Shows the mask in a plot
        """
        fig, ax = plt.subplots()
        plt.plot(self._cen["x"], self._cen["y"], "o", figure=fig)
        plt.show()

def visualise_measurement(res):  
    fig, axs = plt.subplots(2, 3)

    axs[0, 0].plot(res["p_inp"]["x"], res["p_inp"]["y"], "r,", label="input")
    axs[0, 0].plot(res["p_msk"]["x"], res["p_msk"]["y"], "b.", label="mask")
    axs[0, 0].set_title("Input x-y")
    axs[0, 0].set_xlabel("x (mm)")
    axs[0, 0].set_ylabel("y (mm)")
    axs[0, 0].set_aspect('equal', 'datalim')

    axs[0, 1].plot(res["p_inp"]["x"], 1000 * res["p_inp"]["xp"], "r,", label="input")
    axs[0, 1].plot(res["p_msk"]["x"], 1000 * res["p_msk"]["xp"], "b.", label="mask")
    axs[0, 1].set_title("Input x-x' $\\epsilon = %.3f$"%res["emit_inp_x"])
    axs[0, 1].set_xlabel("x (mm)")
    axs[0, 1].set_ylabel("x' (mrad)")

    axs[0, 2].plot(res["p_inp"]["y"], 1000 * res["p_inp"]["yp"], "r,", label="input")
    axs[0, 2].plot(res["p_msk"]["y"], 1000 * res["p_msk"]["yp"], "b.", label="mask")
    axs[0, 2].set_title("Input y-y' $\\epsilon = %.3f$"%res["emit_inp_y"])
    axs[0, 2].set_xlabel("y (mm)")
    axs[0, 2].set_ylabel("y' (mrad)")

    axs[1, 0].plot(res["p_scr"]["x"], res["p_scr"]["y"], "b,", label="input")
    axs[1, 0].set_title("Screen x-y")
    axs[1, 0].set_xlabel("x (mm)")
    axs[1, 0].set_ylabel("y (mm)")
    axs[1, 0].set_aspect('equal', 'datalim')

    axs[1, 1].plot(res["p_rec"]["x"], 1000 * res["p_rec"]["xp"], "b,", label="recon")
    axs[1, 1].set_title("Reconstruction x-x' $\\epsilon = %.3f$"%res["emit_rec_x"])
    axs[1, 1].set_xlabel("x (mm)")
    axs[1, 1].set_ylabel("x' (mrad)")

    axs[1, 2].plot(res["p_rec"]["y"], 1000 * res["p_rec"]["yp"], "b,", label="recon")
    axs[1, 2].set_title("Reconstruction y-y' $\\epsilon = %.3f$"%res["emit_rec_y"])
    axs[1, 2].set_xlabel("y (mm)")
    axs[1, 2].set_ylabel("y' (mrad)")

    plt.show()

### Beam properties
SIGX = 2.0
SIGXP = 1.0e-3
EMITX = 1.5e-3
##derived
COVXXP = np.sqrt(SIGX**2*SIGXP**2-EMITX**2)
COVMATX = np.array([[SIGX**2, COVXXP],[COVXXP,SIGXP**2]])
# print(COVMATX)
# print(np.sqrt(np.linalg.det(COVMATX)))
NSAMPLE = 1000000
x, xp = np.random.multivariate_normal([0,0], COVMATX, NSAMPLE).T
y, yp = np.random.multivariate_normal([0,0], COVMATX, NSAMPLE).T
p = {"x":x, "y":y, "xp":xp, "yp":yp}

mask = (x**2 + y**2) > 2.5**2
p = {"x":x[mask], "y":y[mask], "xp":xp[mask], "yp":yp[mask]}
# p = {"x":0., "y":0., "xp":0., "yp":2.}

pot = Pepperpot()
res = pot.measurement(p)
visualise_measurement(res)
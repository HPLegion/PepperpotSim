import numpy as np
import matplotlib.pyplot as plt

def emittance(x, xp):
    return np.sqrt(np.linalg.det(np.cov(x, xp)))

def emittance_corrected(x, xp, L, R):
    v2 = R**2/4
    return np.sqrt(np.linalg.det(np.cov(x, xp)-np.array([[v2,1.e3*v2/L],[1.e3*v2/L,1.e6*v2/L**2]])))

class Pepperpot:
    """
    Simplified simulation of a pepperpot
    """

    def __init__(self, d=2, r=0.045, l=190, n=51):
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
        self._max_ang = 1000 * np.arctan2(self._d/2, self._l)
        self._max_hole_ang = 1000 * np.arctan2(self._r, self._l)

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
        inds_x = np.digitize(p["x"], self._cen["x"][:-1] + self._d/2)
        inds_y = np.digitize(p["y"], self._cen["y"][:-1] + self._d/2)
        x_cen = self._cen["x"][inds_x]
        y_cen = self._cen["y"][inds_y]
        # x_cen = (p["x"]//self._d) * self._d
        # y_cen = (p["y"]//self._d) * self._d
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
        p_out["x"] = p["x"] + self._l*np.tan(p["xp"]/1000)
        p_out["y"] = p["y"] + self._l*np.tan(p["yp"]/1000)
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
        p_out["xp"] = 1000 * np.arctan2(p["x"]-cen["x"], self._l)
        p_out["yp"] = 1000 * np.arctan2(p["y"]-cen["y"], self._l)
        return p_out

    def measure(self, p_inp, input_emit=False):
        p_msk = self.mask_particles(p_inp)
        p_scr = self.project_on_screen(p_msk)
        p_rec = self.reconstruct_phase_space(p_scr)
        if input_emit:
            emit_inp_x = emittance(p_inp["x"], p_inp["xp"])
            emit_inp_y = emittance(p_inp["y"], p_inp["yp"])
        else:
            emit_inp_x = -1
            emit_inp_y = -1
        emit_rec_x = emittance(p_rec["x"], p_rec["xp"])
        emit_rec_y = emittance(p_rec["y"], p_rec["yp"])
        emit_rec_x = emittance_corrected(p_rec["x"], p_rec["xp"], self._l, self._r)
        emit_rec_y = emittance_corrected(p_rec["y"], p_rec["yp"], self._l, self._r)
        res = {"p_inp":p_inp, "p_msk":p_msk, "p_scr":p_scr, "p_rec": p_rec,
               "emit_inp_x": emit_inp_x, "emit_inp_y": emit_inp_y,
               "emit_rec_x": emit_rec_x, "emit_rec_y": emit_rec_y}
        return res

    def visualise_measurement(self, res):
        def plot_angle_lim(ax):
            ax.axhline(self._max_ang, color="tab:gray", ls="--")
            ax.axhline(self._max_ang - self._max_hole_ang, color="tab:gray", ls="--", lw=1)
            ax.axhline(-self._max_ang, color="tab:gray", ls="--")
            ax.axhline(-self._max_ang + self._max_hole_ang, color="tab:gray", ls="--", lw=1)

        fig, axs = plt.subplots(2, 3, figsize=(16,10), dpi=300)

        axs[0, 0].plot(res["p_inp"]["x"], res["p_inp"]["y"], "r,", label="input")
        axs[0, 0].plot(res["p_msk"]["x"], res["p_msk"]["y"], "b,", label="mask")
        axs[0, 0].set_title("Input x-y $n = %d$"%len(res["p_inp"]["x"]))
        axs[0, 0].set_xlabel("x (mm)")
        axs[0, 0].set_ylabel("y (mm)")
        axs[0, 0].set_aspect('equal', 'datalim')

        axs[0, 1].plot(res["p_inp"]["x"], res["p_inp"]["xp"], "r,", label="input")
        axs[0, 1].plot(res["p_msk"]["x"], res["p_msk"]["xp"], "b,", label="mask")
        plot_angle_lim(axs[0, 1])
        axs[0, 1].set_title("Input x-x' $\\epsilon = %.3f$"%res["emit_inp_x"])
        axs[0, 1].set_xlabel("x (mm)")
        axs[0, 1].set_ylabel("x' (mrad)")

        axs[0, 2].plot(res["p_inp"]["y"], res["p_inp"]["yp"], "r,", label="input")
        axs[0, 2].plot(res["p_msk"]["y"], res["p_msk"]["yp"], "b,", label="mask")
        plot_angle_lim(axs[0, 2])
        axs[0, 2].set_title("Input y-y' $\\epsilon = %.3f$"%res["emit_inp_y"])
        axs[0, 2].set_xlabel("y (mm)")
        axs[0, 2].set_ylabel("y' (mrad)")

        axs[1, 0].plot(res["p_scr"]["x"], res["p_scr"]["y"], "b,", label="input")
        axs[1, 0].set_title("Screen x-y $n = %d$"%len(res["p_scr"]["x"]))
        axs[1, 0].set_xlabel("x (mm)")
        axs[1, 0].set_ylabel("y (mm)")
        axs[1, 0].set_aspect('equal', "datalim")

        xbins = self._cen["x"] + self._d/2
        ybins = np.linspace(axs[0, 1].get_ylim()[0], axs[0, 1].get_ylim()[1], 100)
        axs[1, 1].hist2d(res["p_rec"]["x"], res["p_rec"]["xp"],
                         bins=[xbins, ybins], cmap="plasma", cmin=1)
        axs[1, 1].plot(res["p_rec"]["x"], res["p_rec"]["xp"], "b,", label="recon")
        plot_angle_lim(axs[1, 1])
        axs[1, 1].set_title("Reconstruction x-x' $\\epsilon = %.3f$"%res["emit_rec_x"])
        axs[1, 1].set_xlabel("x (mm)")
        axs[1, 1].set_ylabel("x' (mrad)")
        axs[1, 1].set_xlim(axs[0, 1].get_xlim())
        axs[1, 1].set_ylim(axs[0, 1].get_ylim())

        xbins = self._cen["y"] + self._d/2
        ybins = np.linspace(axs[0, 1].get_ylim()[0], axs[0, 1].get_ylim()[1], 100)
        axs[1, 2].hist2d(res["p_rec"]["y"], res["p_rec"]["yp"],
                         bins=[xbins, ybins], cmap="plasma", cmin=1)
        axs[1, 2].plot(res["p_rec"]["y"], res["p_rec"]["yp"], "b,", label="recon")
        plot_angle_lim(axs[1, 2])
        axs[1, 2].set_title("Reconstruction y-y' $\\epsilon = %.3f$"%res["emit_rec_y"])
        axs[1, 2].set_xlabel("y (mm)")
        axs[1, 2].set_ylabel("y' (mrad)")
        axs[1, 2].set_xlim(axs[0, 2].get_xlim())
        axs[1, 2].set_ylim(axs[0, 2].get_ylim())

        plt.show()

    def show_mask(self):
        """
        Shows the mask in a plot
        """
        fig, ax = plt.subplots()
        plt.plot(self._cen["x"], self._cen["y"], "o", figure=fig)
        plt.show()

    
def generate_gaussian_beam_1D(n, sigx, sigxp, emitx):
    covxxp = np.sqrt(sigx**2 * sigxp**2 - emitx**2)
    covmat = np.array([[sigx**2, covxxp], [covxxp, sigxp**2]])
    x, xp = np.random.multivariate_normal([0,0], covmat, n).T
    return x, xp

def generate_gaussian_beam_2D(n, sigx, sigxp, emitx, sigy, sigyp, emity):
    x, xp = generate_gaussian_beam_1D(n, sigx, sigxp, emitx)
    y, yp = generate_gaussian_beam_1D(n, sigy, sigyp, emity)
    return {"x":x, "y":y, "xp":xp, "yp":yp}
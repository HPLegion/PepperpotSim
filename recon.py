import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import scipy.ndimage, scipy.signal
import imageio


# class CallbackAttribute():
#     def __init__(self, value, subscriber=None):
#         self._value = value
#         if subscriber:
#             self._subscribers = [subscriber]
#         else:
#             self._subscribers = []

#     def __instancecheck__(self, instance):
#         return isinstance(instance, type(self._value))

#     def __get__(self, instance, owner):
#         return self._value

#     def __set__(self, instance, value):
#         self._value = value
#         for callback in self._subscribers:
#             callback()

#     def add_subscriber(self, subscriber):
#         self._subscribers.append(subscriber)

class PhaseSpaceReconstructor:
    def __init__(self):
        self.image_raw = None
        self.image_filtered = None
        self.image_scale_x = 0.00004 #m/pix
        self.image_scale_y = 0.00004
        self.image_roi = None

        self.filter_offset = True
        self.filter_median = True
        self.filter_clip = True
        self.filter_offset_value = 0
        self.filter_median_value = 1
        self.filter_clip_value_min = 0
        self.filter_clip_value_max = None

        self.peakfind_distance_x = 10
        self.peakfind_distance_y = 10
        self.peakfind_height_x = 10
        self.peakfind_height_y = 10

        self.mask_screen_distance = .190
        self.mask_hole_radius = 0.000045
        self.mask_hole_spacing_x = .002
        self.mask_hole_spacing_y = .002

        self.results = dict()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        update_filters_subs = ["filter_offset", "filter_median", "filter_clip",
                               "filter_offset_value", "filter_median_value",
                               "filter_clip_value_min", "filter_clip_value_max"]

        update_reconstruction_subs = ["self.peakfind_distance_x", "peakfind_distance_y",
                                      "peakfind_height_x", "peakfind_height_y", "mask_screen_distance",
                                      "mask_hole_radius", "mask_hole_spacing_x", "mask_hole_spacing_y",
                                      "image_scale_x", "image_scale_x"]
        if name in update_filters_subs:
            self.update_filters()
        if name in update_reconstruction_subs:
            self.update_reconstruction()

    def load_image(self, file):
        self.image_raw = imageio.imread(file)
        self.image_raw_marginal_x = self.image_raw.mean(0)
        self.image_raw_marginal_y = self.image_raw.mean(1)
        self.update_filters()

    def update_filters(self):
        if self.image_raw is None:
            return
        img = self.image_raw.copy()
        if self.filter_offset:
            img = img - self.filter_offset_value
        if self.filter_median:
            for _ in range(self.filter_median_value):
                img = scipy.ndimage.median_filter(img, size=2)
        if self.filter_clip:
            img = np.clip(img, self.filter_clip_value_min, self.filter_clip_value_max)
        self.image_filtered = img
        self.image_filtered_marginal_x = self.image_filtered.mean(0)
        self.image_filtered_marginal_y = self.image_filtered.mean(1)
        self.update_reconstruction()

    def _update_peaks(self):
        self.peaks_x, _ = scipy.signal.find_peaks(self.image_filtered_marginal_x,
                                                  distance=self.peakfind_distance_x,
                                                  height=self.peakfind_height_x)
        self.peaks_y, _ = scipy.signal.find_peaks(self.image_filtered_marginal_y,
                                                  distance=self.peakfind_distance_y,
                                                  height=self.peakfind_height_y)
    def _update_separators(self):
        roi = {}
        seps = (self.peaks_x[:-1] + self.peaks_x[1:])/2
        seps = np.insert(seps, 0, 2*seps[0]-seps[1])
        seps = np.append(seps, 2*seps[-1]-seps[-2])
        seps = np.clip(seps, 0, self.image_filtered_marginal_x.size-1)
        seps = np.floor(seps).astype(int)
        roi["min_x"] = seps[0].clip(0, None)
        roi["max_x"] = seps[-1].clip(None, self.image_filtered.shape[1]+1)
        self.seperators_x = seps - seps[0]

        seps = (self.peaks_y[:-1] + self.peaks_y[1:])/2
        seps = np.insert(seps, 0, 2*seps[0]-seps[1])
        seps = np.append(seps, 2*seps[-1]-seps[-2])
        seps = np.clip(seps, 0, self.image_filtered_marginal_y.size-1)
        seps = np.floor(seps).astype(int)
        roi["min_y"] = seps[0].clip(0, None)
        roi["max_y"] = seps[-1].clip(None, self.image_filtered.shape[0]+1)
        self.seperators_y = seps - seps[0]

        self.image_roi = roi

    def _update_masked_image(self):
        self.image_masked = self.image_filtered[self.image_roi["min_y"]:self.image_roi["max_y"],
                                                self.image_roi["min_x"]:self.image_roi["max_x"]]
        self.image_masked_marginal_x = self.image_masked.mean(0).clip(0, None)
        self.image_masked_marginal_y = self.image_masked.mean(1).clip(0, None)

    def _update_hole_positions(self):
        self.holes_x = np.arange(self.peaks_x.size) * self.mask_hole_spacing_x
        self.holes_y = np.arange(self.peaks_y.size) * self.mask_hole_spacing_y

    def _update_maps(self):
        self.map_x = np.zeros_like(self.image_masked_marginal_x)
        for j in range(self.seperators_x.size-1):
            self.map_x[self.seperators_x[j]:self.seperators_x[j+1]] = self.holes_x[j]
        self.map_xp = (np.arange(self.image_masked_marginal_x.size)*self.image_scale_x - self.map_x) / self.mask_screen_distance

        self.map_y = np.zeros_like(self.image_masked_marginal_y)
        for j in range(self.seperators_y.size-1):
            self.map_y[self.seperators_y[j]:self.seperators_y[j+1]] = self.holes_y[j]
        self.map_yp = (np.arange(self.image_masked_marginal_y.size)*self.image_scale_y - self.map_y) / self.mask_screen_distance

        self.map_x = self.map_x - self.map_x.mean()
        self.map_xp = self.map_xp - self.map_xp.mean()
        self.map_y = self.map_y - self.map_y.mean()
        self.map_yp = self.map_yp - self.map_yp.mean()


    def _update_results(self):
        results = {}

        var = self.mask_hole_radius**2 / 4
        length = self.mask_screen_distance
        correct = np.array([[var, var/length], [var/length, var/length**2]])

        xcov = np.cov(self.map_x, self.map_xp, aweights=self.image_masked_marginal_x)
        emittance_x = np.sqrt(np.linalg.det(xcov))
        emittance_x_corrected = np.sqrt(np.linalg.det(xcov-correct))
        results["emittance_x"] = emittance_x
        results["emittance_x_corrected"] = emittance_x_corrected
        results["alpha_x"] = -1*xcov[0, 1]/emittance_x
        results["beta_x"] = xcov[0, 0]/emittance_x
        results["gamma_x"] = xcov[1, 1]/emittance_x

        ycov = np.cov(self.map_y, self.map_yp, aweights=self.image_masked_marginal_y)
        emittance_y = np.sqrt(np.linalg.det(ycov))
        emittance_y_corrected = np.sqrt(np.linalg.det(ycov-correct))
        results["emittance_y"] = emittance_y
        results["emittance_y_corrected"] = emittance_y_corrected
        results["alpha_y"] = -1*ycov[0, 1]/emittance_y
        results["beta_y"] = ycov[0, 0]/emittance_y
        results["gamma_y"] = ycov[1, 1]/emittance_y

        self.results.update(results)

    def update_reconstruction(self):
        if self.image_filtered is None:
            return
        self._update_peaks()
        if self.peaks_x.size < 3 and self.peaks_y.size < 3:
            return
        self._update_separators()
        self._update_masked_image()
        self._update_hole_positions()
        self._update_maps()
        self._update_results()

    def plot_input(self, **kwargs):
        plt.figure(figsize=(15,10))
        plt.imshow(self.image_raw, cmap="plasma", **kwargs)
        plt.title("Raw image")
        # plt.colorbar()
        for x in self.seperators_x:
            plt.axvline(self.image_roi["min_x"] + x)
        for y in self.seperators_y:
            plt.axhline(self.image_roi["min_y"] + y)
        plt.show()

    def plot_filtered_roi(self, **kwargs):
        plt.figure(figsize=(15,10))
        plt.imshow(self.image_masked, cmap="plasma", **kwargs)
        plt.title("Filtered ROI")
        # plt.colorbar()
        for x in self.seperators_x:
            plt.axvline(x)
        for y in self.seperators_y:
            plt.axhline(y)
        plt.show()

    def plot_reconst(self, **kwargs):
        plt.figure()
        fig, axs = plt.subplots(1,2, figsize=(20,10))

        xbins = np.sort(np.unique(self.map_x-self.mask_hole_spacing_x/2))
        xbins = np.append(xbins, xbins[-1] + self.mask_hole_spacing_x)
        ybins = np.arange(self.map_xp.min(), self.map_xp.max(),
                          self.image_scale_x/self.mask_screen_distance)
        axs[0].hist2d(self.map_x, self.map_xp, weights=self.image_masked_marginal_x,
                      bins=([xbins, ybins]), cmin=1, cmap="plasma")
        axs[0].set_xlabel("x (m)")
        axs[0].set_ylabel("x' (rad)")
        axs[0].set_title("x-x' Reconstruction")

        xbins = np.sort(np.unique(self.map_y-self.mask_hole_spacing_y/2))
        xbins = np.append(xbins, xbins[-1] + self.mask_hole_spacing_y)
        ybins = np.arange(self.map_yp.min(), self.map_yp.max(),
                          self.image_scale_y/self.mask_screen_distance)
        axs[1].hist2d(self.map_y, self.map_yp, weights=self.image_masked_marginal_y,
                      bins=([xbins, ybins]), cmin=1, cmap="plasma")
        axs[1].set_xlabel("y (m)")
        axs[1].set_ylabel("y' (rad)")
        axs[1].set_title("y-y' Reconstruction")

        plt.show()

    def plot_marginals(self, **kwargs):
        plt.figure()
        fig, axs = plt.subplots(2,1, figsize=(10,10))


        axs[0].set_xlabel("x (pix)")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].set_title("x mean intensity")
        axs[0].plot(self.peaks_x, self.image_filtered_marginal_x[self.peaks_x], "o", label="peaks")
        axs[0].plot(self.image_raw_marginal_x, c="r", lw=0.5, label="raw")
        axs[0].plot(self.image_filtered_marginal_x, c="b", lw=0.5, label="filtered")
        axs[0].legend()


        axs[1].set_xlabel("y (pix)")
        axs[1].set_ylabel("Intensity (a.u.)")
        axs[1].set_title("y mean intensity")
        axs[1].plot(self.peaks_y, self.image_filtered_marginal_y[self.peaks_y], "o", label="peaks")
        axs[1].plot(self.image_raw_marginal_y, c="r", lw=0.5, label="raw")
        axs[1].plot(self.image_filtered_marginal_y, c="b", lw=0.5, label="filtered")
        axs[1].legend()

        plt.show()

    def print_results(self):
        for key, item in self.results.items():
            print(key, "=", item)
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from recon_gui import Ui_MainWindow
from recon import PhaseSpaceReconstructor


class ReconForm():
    """
    Class tying together the gui and reconstruction logic
    """
    def __init__(self):
        # Object performing the actual reconstruction
        self.psr = PhaseSpaceReconstructor(auto_update=False)

        # Create UI
        self.main_window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)

        # Prepare plotting
        self.ui.canvas_input.ax = self.ui.canvas_input.fig.subplots()
        self.ui.canvas_roi.ax = self.ui.canvas_roi.fig.subplots()
        self.ui.canvas_recon.ax = self.ui.canvas_recon.fig.subplots(1, 2)
        self.ui.canvas_marginals.ax = self.ui.canvas_marginals.fig.subplots(1, 2)
        
        # Connect load image button
        self.ui.btn_load_image.clicked.connect(self.load_image)

        # Connect device dimension controls
        self.ui.dsb_mask_screen_distance.valueChanged.connect(self.update_psr_device)
        self.ui.dsb_mask_hole_radius.valueChanged.connect(self.update_psr_device)
        self.ui.dsb_mask_hole_spacing_x.valueChanged.connect(self.update_psr_device)
        self.ui.dsb_mask_hole_spacing_y.valueChanged.connect(self.update_psr_device)
        self.ui.dsb_image_scale_x.valueChanged.connect(self.update_psr_device)
        self.ui.dsb_image_scale_y.valueChanged.connect(self.update_psr_device)

        # Connect filter settings
        self.ui.cb_offset_filter.stateChanged.connect(self.update_psr_filter)
        self.ui.cb_median_filter.stateChanged.connect(self.update_psr_filter)
        self.ui.cb_clip_filter.stateChanged.connect(self.update_psr_filter)
        self.ui.dsb_offset_filter.valueChanged.connect(self.update_psr_filter)
        self.ui.sb_median_filter.valueChanged.connect(self.update_psr_filter)
        self.ui.dsb_clip_filter_min.valueChanged.connect(self.update_psr_filter)
        self.ui.dsb_clip_filter_max.valueChanged.connect(self.update_psr_filter)        

        # Connect peak detection settings
        self.ui.dsb_peak_distance_x.valueChanged.connect(self.update_psr_peak_detection)        
        self.ui.dsb_peak_distance_y.valueChanged.connect(self.update_psr_peak_detection)        
        self.ui.dsb_peak_height_x.valueChanged.connect(self.update_psr_peak_detection)        
        self.ui.dsb_peak_height_y.valueChanged.connect(self.update_psr_peak_detection)        
        
        # Show window
        self.main_window.show()

    def update_ui_plots(self):
        self.ui.canvas_input.ax.clear()
        self.psr.plot_input(self.ui.canvas_input.ax)
        self.ui.canvas_input.draw()

        self.ui.canvas_roi.ax.clear()
        self.psr.plot_filtered_roi(self.ui.canvas_roi.ax)
        self.ui.canvas_roi.draw()

        self.ui.canvas_recon.ax[0].clear()
        self.ui.canvas_recon.ax[1].clear()
        self.psr.plot_reconst(self.ui.canvas_recon.ax)
        self.ui.canvas_recon.draw()

        self.ui.canvas_marginals.ax[0].clear()
        self.ui.canvas_marginals.ax[1].clear()
        self.psr.plot_marginals(self.ui.canvas_marginals.ax)
        self.ui.canvas_marginals.draw()


    def update_ui_results(self):
        self.ui.dsb_result_emittance_x.setValue(1.0e6 * self.psr.results.get("emittance_x", 0))
        self.ui.dsb_result_emittance_y.setValue(1.0e6 * self.psr.results.get("emittance_y", 0))
        self.ui.dsb_result_emittance_corrected_x.setValue(1.0e6 * self.psr.results.get("emittance_x_corrected", 0))
        self.ui.dsb_result_emittance_corrected_y.setValue(1.0e6 * self.psr.results.get("emittance_y_corrected", 0))
        self.ui.dsb_result_alpha_x.setValue(self.psr.results.get("alpha_x", 0))
        self.ui.dsb_result_alpha_y.setValue(self.psr.results.get("alpha_y", 0))
        self.ui.dsb_result_beta_x.setValue(self.psr.results.get("beta_x", 0))
        self.ui.dsb_result_beta_y.setValue(self.psr.results.get("beta_y", 0))
        self.ui.dsb_result_gamma_x.setValue(self.psr.results.get("gamma_x", 0))
        self.ui.dsb_result_gamma_y.setValue(self.psr.results.get("gamma_y", 0))
        self.update_ui_plots()

    def update_psr_device(self):
        self.psr.mask_screen_distance = 1.0e-3 * self.ui.dsb_mask_screen_distance.value()
        self.psr.mask_hole_radius = 1.0e-6 * self.ui.dsb_mask_hole_radius.value()
        self.psr.mask_hole_spacing_x = 1.0e-3 * self.ui.dsb_mask_hole_spacing_x.value()
        self.psr.mask_hole_spacing_y = 1.0e-3 * self.ui.dsb_mask_hole_spacing_y.value()
        self.psr.image_scale_x = 1.0e-6 * self.ui.dsb_image_scale_x.value()
        self.psr.image_scale_y = 1.0e-6 * self.ui.dsb_image_scale_y.value()
        self.psr.update_reconstruction()
        self.update_ui_results()

    def update_psr_peak_detection(self):
        self.psr.peakfind_distance_x = self.ui.dsb_peak_distance_x.value()
        self.psr.peakfind_distance_y = self.ui.dsb_peak_distance_y.value()
        self.psr.peakfind_height_x = self.ui.dsb_peak_height_x.value()
        self.psr.peakfind_height_y = self.ui.dsb_peak_height_y.value()
        self.psr.update_reconstruction()
        self.update_ui_results()

    def update_psr_filter(self):
        self.psr.filter_offset = self.ui.cb_offset_filter.isChecked()
        self.psr.filter_median = self.ui.cb_median_filter.isChecked()
        self.psr.filter_clip = self.ui.cb_clip_filter.isChecked()
        self.psr.filter_offset_value = self.ui.dsb_offset_filter.value()
        self.psr.filter_median_value = self.ui.sb_median_filter.value()
        self.psr.filter_clip_value_min = self.ui.dsb_clip_filter_min.value()
        val = self.ui.dsb_clip_filter_max.value()
        if val == 0:
            self.psr.filter_clip_value_max = None
        else:
            self.psr.filter_clip_value_max = val
        self.psr.update_filters()
        self.update_ui_results()

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_window, "Open image for analysis")
        try:
            self.psr.load_image(path)
        except:
            QtWidgets.QMessageBox.warning(self.main_window, "Error!", "Failed to load image,\n please choose a valid file.")
        self.update_psr_device()
        self.update_psr_filter()
        self.update_psr_peak_detection()
        self.ui.group_device_dimensions.setEnabled(True)
        self.ui.group_filter_settings.setEnabled(True)
        self.ui.group_peak_detection.setEnabled(True)
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    form = ReconForm()
    sys.exit(app.exec_())
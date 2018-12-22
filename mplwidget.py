from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure


class MplWidget(FigureCanvas):
    def __init__(self, parent):
        self.fig = Figure()
        super(MplWidget, self).__init__(self.fig)
        self.setParent(parent)

    def sizeHint(self):
        return QSize(*self.get_width_height())

    def minimumSizeHint(self):
        return QSize(10, 10)



# from PyQt5.QtCore import QSize
# from PyQt5.QtWidgets import QSizePolicy
# from matplotlib.backends.backend_qt5agg import FigureCanvas
# from matplotlib.figure import Figure


# class MatplotlibWidget(FigureCanvas):
#     def __init__(self, parent=None, title='', xlabel='', ylabel='',
#                  xlim=None, ylim=None, xscale='linear', yscale='linear',
#                  width=4, height=3, dpi=100):
#         self.figure = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = self.figure.add_subplot(111)
#         self.axes.set_title(title)
#         self.axes.set_xlabel(xlabel)
#         self.axes.set_ylabel(ylabel)
#         if xscale is not None:
#             self.axes.set_xscale(xscale)
#         if yscale is not None:
#             self.axes.set_yscale(yscale)
#         if xlim is not None:
#             self.axes.set_xlim(*xlim)
#         if ylim is not None:
#             self.axes.set_ylim(*ylim)

#         super(MatplotlibWidget, self).__init__(self.figure)
#         self.setParent(parent)
#         super(MatplotlibWidget, self).setSizePolicy(
#             QSizePolicy.Expanding, QSizePolicy.Expanding)
#         super(MatplotlibWidget, self).updateGeometry()

#     def sizeHint(self):
#         return QSize(*self.get_width_height())

#     def minimumSizeHint(self):
#         return QSize(10, 10)

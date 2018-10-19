import sys

from PyQt5.QtCore import QPoint, Qt, QRect, QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPushButton, QRadioButton, QGridLayout, \
    QRubberBand, QFileDialog, QMessageBox

import numpy as np
import imageio
from shutil import copyfile

import matplotlib.pyplot as plt
from skimage.restoration import inpaint

debug = False

class ImageProcessor(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Inpaint application')

        self.image1 = Image()
        self.image1.setMinimumWidth(250)

        upload_button = QPushButton('upload')
        upload_button.clicked.connect(self.upload)

        transform_button = QPushButton('transform')
        transform_button.clicked.connect(self.transform)

        grid = QGridLayout()
        grid.addWidget(self.image1, 1, 0)
        grid.addWidget(upload_button, 2, 0)
        grid.addWidget(transform_button, 2, 3)
        self.setLayout(grid)

        self.show()

    def call_inpaint(self, coordinates):
        image_orig = imageio.imread(self.image1.image_path)
        mask = np.zeros(image_orig.shape[:-1])
        mask[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]] = 1

        image_defect = image_orig.copy()

        for layer in range(image_defect.shape[-1]):
            image_defect[np.where(mask)] = 0

        image_result = inpaint.inpaint_biharmonic(image_defect, mask, multichannel=True)

        if debug:
            fig, axes = plt.subplots(ncols=2, nrows=2)
            ax = axes.ravel()

            ax[0].set_title('Original image')
            ax[0].imshow(image_orig)

            ax[1].set_title('Mask')
            ax[1].imshow(mask, cmap=plt.cm.gray)

            ax[2].set_title('Defected image')
            ax[2].imshow(image_defect)

            ax[3].set_title('Inpainted image')
            ax[3].imshow(image_result)

            for a in ax:
                a.axis('off')

            fig.tight_layout()
            plt.show()

        imageio.imwrite(self.image1.image_path, image_result)
        self.image1.selection.hide()


    def upload(self):
        # taken from https://pythonspot.com/en/pyqt5-file-dialog/
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, 'Open file', options=options)
        if file:
            height, width, _ = imageio.imread(file).shape
            self.image1.setMinimumSize(width, height)
            self.image1.image_path = file.rsplit('.', 1)[0]+'_out.'+file.rsplit('.', 1)[1]
            copyfile(file, self.image1.image_path)
            self.image1.setPixmap(QPixmap(self.image1.image_path))


    def transform(self):
        if self.image1.image_path is None:
            QMessageBox.information(self, 'Error', "Please upload an image.", QMessageBox.Ok, QMessageBox.Ok)
        elif (self.image1.selection_origin is None or self.image1.selection_end is None) and self.image1.image_path is not None:
            QMessageBox.information(self, 'Error', "Please select a rectangular area.", QMessageBox.Ok, QMessageBox.Ok)
        else:
            coordinates = self.get_coordinates(self.image1.selection_origin, self.image1.selection_end)
            self.call_inpaint(coordinates)
            self.image1.setPixmap(QPixmap(self.image1.image_path))


    def get_coordinates(self, origin, end):
        origin = (origin.x(), origin.y())
        end = (end.x(), end.y())
        return origin, end


class Image(QLabel):
    # taken from https://wiki.python.org/moin/PyQt/Selecting%20a%20region%20of%20a%20widget
    def __init__(self, parent=None):
        QLabel.__init__(self, parent)
        self.selection = QRubberBand(QRubberBand.Rectangle, self)
        self.selection_origin = None
        self.selection_end = None
        self.image_path = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_origin = QPoint(event.pos())
            self.selection.setGeometry(QRect(self.selection_origin, QSize()))
            self.selection.show()

    def mouseMoveEvent(self, event):
        if not self.selection_origin.isNull():
            self.selection.setGeometry(QRect(self.selection_origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_end = QPoint(event.pos())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    imageprocessor = ImageProcessor()
    sys.exit(app.exec_())

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt6 UI code generator 6.5.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1205, 808)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tableView = QtWidgets.QTableView(parent=self.centralwidget)
        self.tableView.setSortingEnabled(True)
        self.tableView.setObjectName("tableView")
        self.gridLayout_5.addWidget(self.tableView, 0, 0, 1, 2)
        self.textBrowserInfo = QtWidgets.QTextBrowser(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.textBrowserInfo.setFont(font)
        self.textBrowserInfo.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.textBrowserInfo.setObjectName("textBrowserInfo")
        self.gridLayout_5.addWidget(self.textBrowserInfo, 1, 1, 1, 1)
        self.textBrowser = QtWidgets.QTextBrowser(parent=self.centralwidget)
        self.textBrowser.setObjectName("textBrowser")
        self.gridLayout_5.addWidget(self.textBrowser, 1, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.comboBoxAtrybutClass = QtWidgets.QComboBox(parent=self.centralwidget)
        self.comboBoxAtrybutClass.setObjectName("comboBoxAtrybutClass")
        self.gridLayout_4.addWidget(self.comboBoxAtrybutClass, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 0, 1, 1)
        self.pushButtonClass = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonClass.setObjectName("pushButtonClass")
        self.gridLayout_4.addWidget(self.pushButtonClass, 1, 1, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 2, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.comboBoxAtrybut2 = QtWidgets.QComboBox(parent=self.centralwidget)
        self.comboBoxAtrybut2.setObjectName("comboBoxAtrybut2")
        self.gridLayout_2.addWidget(self.comboBoxAtrybut2, 1, 1, 1, 1)
        self.pushButtonKoorelacja = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonKoorelacja.setObjectName("pushButtonKoorelacja")
        self.gridLayout_2.addWidget(self.pushButtonKoorelacja, 1, 2, 1, 1)
        self.comboBoxAtrybut1 = QtWidgets.QComboBox(parent=self.centralwidget)
        self.comboBoxAtrybut1.setObjectName("comboBoxAtrybut1")
        self.gridLayout_2.addWidget(self.comboBoxAtrybut1, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_2, 3, 0, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)
        self.pushButtonPorownaj = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonPorownaj.setObjectName("pushButtonPorownaj")
        self.gridLayout_3.addWidget(self.pushButtonPorownaj, 2, 0, 1, 1)
        self.pushButtonDystrybucja = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonDystrybucja.setObjectName("pushButtonDystrybucja")
        self.gridLayout_3.addWidget(self.pushButtonDystrybucja, 2, 1, 1, 1)
        self.pushButtonHeatmap = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonHeatmap.setObjectName("pushButtonHeatmap")
        self.gridLayout_3.addWidget(self.pushButtonHeatmap, 2, 2, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_3, 3, 1, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButtonCalcChecked = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonCalcChecked.setObjectName("pushButtonCalcChecked")
        self.gridLayout.addWidget(self.pushButtonCalcChecked, 1, 5, 1, 1)
        self.checkBoxStd = QtWidgets.QCheckBox(parent=self.centralwidget)
        self.checkBoxStd.setEnabled(True)
        self.checkBoxStd.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkBoxStd.setAutoFillBackground(False)
        self.checkBoxStd.setText("")
        self.checkBoxStd.setObjectName("checkBoxStd")
        self.gridLayout.addWidget(self.checkBoxStd, 1, 2, 1, 1)
        self.checkBoxMean = QtWidgets.QCheckBox(parent=self.centralwidget)
        self.checkBoxMean.setEnabled(True)
        self.checkBoxMean.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkBoxMean.setAutoFillBackground(False)
        self.checkBoxMean.setText("")
        self.checkBoxMean.setObjectName("checkBoxMean")
        self.gridLayout.addWidget(self.checkBoxMean, 1, 4, 1, 1)
        self.checkBoxMdn = QtWidgets.QCheckBox(parent=self.centralwidget)
        self.checkBoxMdn.setEnabled(True)
        self.checkBoxMdn.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkBoxMdn.setAutoFillBackground(False)
        self.checkBoxMdn.setText("")
        self.checkBoxMdn.setObjectName("checkBoxMdn")
        self.gridLayout.addWidget(self.checkBoxMdn, 1, 3, 1, 1)
        self.pushButtonStd = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonStd.setObjectName("pushButtonStd")
        self.gridLayout.addWidget(self.pushButtonStd, 2, 2, 1, 1)
        self.pushButtonMaximum = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonMaximum.setObjectName("pushButtonMaximum")
        self.gridLayout.addWidget(self.pushButtonMaximum, 2, 1, 1, 1)
        self.pushButtonMean = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonMean.setObjectName("pushButtonMean")
        self.gridLayout.addWidget(self.pushButtonMean, 2, 4, 1, 1)
        self.pushButtonMedian = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonMedian.setObjectName("pushButtonMedian")
        self.gridLayout.addWidget(self.pushButtonMedian, 2, 3, 1, 1)
        self.pushButtonClear = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonClear.setObjectName("pushButtonClear")
        self.gridLayout.addWidget(self.pushButtonClear, 2, 5, 1, 1)
        self.checkBoxMin = QtWidgets.QCheckBox(parent=self.centralwidget)
        self.checkBoxMin.setEnabled(True)
        self.checkBoxMin.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkBoxMin.setAutoFillBackground(False)
        self.checkBoxMin.setText("")
        self.checkBoxMin.setObjectName("checkBoxMin")
        self.gridLayout.addWidget(self.checkBoxMin, 1, 0, 1, 1)
        self.pushButtonMinimum = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButtonMinimum.setObjectName("pushButtonMinimum")
        self.gridLayout.addWidget(self.pushButtonMinimum, 2, 0, 1, 1)
        self.checkBoxMax = QtWidgets.QCheckBox(parent=self.centralwidget)
        self.checkBoxMax.setEnabled(True)
        self.checkBoxMax.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkBoxMax.setAutoFillBackground(False)
        self.checkBoxMax.setText("")
        self.checkBoxMax.setObjectName("checkBoxMax")
        self.gridLayout.addWidget(self.checkBoxMax, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout, 4, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1205, 21))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(parent=self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menuZmie_motyw = QtWidgets.QMenu(parent=self.menuMenu)
        self.menuZmie_motyw.setObjectName("menuZmie_motyw")
        self.menuWczytaj_z_CSV = QtWidgets.QMenu(parent=self.menuMenu)
        self.menuWczytaj_z_CSV.setObjectName("menuWczytaj_z_CSV")
        self.menuSave_to = QtWidgets.QMenu(parent=self.menuMenu)
        self.menuSave_to.setObjectName("menuSave_to")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionGenerateResultsPDF = QtGui.QAction(parent=MainWindow)
        self.actionGenerateResultsPDF.setObjectName("actionGenerateResultsPDF")
        self.actionWprowadzNagl = QtGui.QAction(parent=MainWindow)
        self.actionWprowadzNagl.setObjectName("actionWprowadzNagl")
        self.actionJasny = QtGui.QAction(parent=MainWindow)
        self.actionJasny.setObjectName("actionJasny")
        self.actionCiemny = QtGui.QAction(parent=MainWindow)
        self.actionCiemny.setObjectName("actionCiemny")
        self.actionWyszukaj = QtGui.QAction(parent=MainWindow)
        self.actionWyszukaj.setObjectName("actionWyszukaj")
        self.actionCSV_load = QtGui.QAction(parent=MainWindow)
        self.actionCSV_load.setObjectName("actionCSV_load")
        self.actionJSON_load = QtGui.QAction(parent=MainWindow)
        self.actionJSON_load.setObjectName("actionJSON_load")
        self.actionCSV_save = QtGui.QAction(parent=MainWindow)
        self.actionCSV_save.setObjectName("actionCSV_save")
        self.actionJSON_save = QtGui.QAction(parent=MainWindow)
        self.actionJSON_save.setObjectName("actionJSON_save")
        self.menuZmie_motyw.addAction(self.actionJasny)
        self.menuZmie_motyw.addAction(self.actionCiemny)
        self.menuWczytaj_z_CSV.addSeparator()
        self.menuWczytaj_z_CSV.addAction(self.actionCSV_load)
        self.menuWczytaj_z_CSV.addAction(self.actionJSON_load)
        self.menuSave_to.addAction(self.actionCSV_save)
        self.menuSave_to.addAction(self.actionJSON_save)
        self.menuMenu.addAction(self.menuWczytaj_z_CSV.menuAction())
        self.menuMenu.addAction(self.menuSave_to.menuAction())
        self.menuMenu.addAction(self.actionWprowadzNagl)
        self.menuMenu.addAction(self.actionGenerateResultsPDF)
        self.menuMenu.addAction(self.menuZmie_motyw.menuAction())
        self.menuMenu.addAction(self.actionWyszukaj)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PyNalise"))
        self.textBrowserInfo.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:14pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"tw-target-text-container\"></a><span style=\" font-family:\'inherit\'; font-size:28px;\">I</span><span style=\" font-family:\'inherit\'; font-size:28px;\">nformation window</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "Classyfication:"))
        self.pushButtonClass.setText(_translate("MainWindow", "Classificate"))
        self.pushButtonKoorelacja.setText(_translate("MainWindow", "Count Coorelation"))
        self.label.setText(_translate("MainWindow", "Coorelation:"))
        self.label_3.setText(_translate("MainWindow", "Plots:"))
        self.pushButtonPorownaj.setText(_translate("MainWindow", "Generate Count Plot"))
        self.pushButtonDystrybucja.setText(_translate("MainWindow", "Generate Distribution Plot"))
        self.pushButtonHeatmap.setText(_translate("MainWindow", "Generate Coorelation Plot"))
        self.pushButtonCalcChecked.setText(_translate("MainWindow", "Calculate Selected"))
        self.pushButtonStd.setText(_translate("MainWindow", "Standard Deviation"))
        self.pushButtonMaximum.setText(_translate("MainWindow", "Max"))
        self.pushButtonMean.setText(_translate("MainWindow", "Arithmetic Average"))
        self.pushButtonMedian.setText(_translate("MainWindow", "Median"))
        self.pushButtonClear.setText(_translate("MainWindow", "Clear"))
        self.pushButtonMinimum.setText(_translate("MainWindow", "Min"))
        self.label_2.setText(_translate("MainWindow", "Stats:"))
        self.menuMenu.setTitle(_translate("MainWindow", "File"))
        self.menuZmie_motyw.setTitle(_translate("MainWindow", "Change theme"))
        self.menuWczytaj_z_CSV.setTitle(_translate("MainWindow", "Load From"))
        self.menuSave_to.setTitle(_translate("MainWindow", "Save to"))
        self.actionGenerateResultsPDF.setText(_translate("MainWindow", "Generate PDF"))
        self.actionWprowadzNagl.setText(_translate("MainWindow", "Insert Headers"))
        self.actionJasny.setText(_translate("MainWindow", "Light"))
        self.actionCiemny.setText(_translate("MainWindow", "Dark"))
        self.actionWyszukaj.setText(_translate("MainWindow", "Find"))
        self.actionCSV_load.setText(_translate("MainWindow", "CSV"))
        self.actionJSON_load.setText(_translate("MainWindow", "JSON"))
        self.actionCSV_save.setText(_translate("MainWindow", "CSV"))
        self.actionJSON_save.setText(_translate("MainWindow", "JSON"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())

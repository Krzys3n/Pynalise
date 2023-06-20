# Form implementation generated from reading ui file 'LoginWidget.ui'
#
# Created by: PyQt6 UI code generator 6.5.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QTableView


class LoginWidget(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(294, 311)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(parent=Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.lineEditLogin = QtWidgets.QLineEdit(parent=Form)
        self.lineEditLogin.setObjectName("lineEditLogin")
        self.gridLayout.addWidget(self.lineEditLogin, 0, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.lineEditPassword = QtWidgets.QLineEdit(parent=Form)
        self.lineEditPassword.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.lineEditPassword.setObjectName("lineEditPassword")
        self.gridLayout.addWidget(self.lineEditPassword, 1, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButtonSignIn = QtWidgets.QPushButton(parent=Form)
        self.pushButtonSignIn.setObjectName("pushButtonSignIn")
        self.horizontalLayout.addWidget(self.pushButtonSignIn)
        self.pushButtonSignUp = QtWidgets.QPushButton(parent=Form)
        self.pushButtonSignUp.setObjectName("pushButtonSignUp")
        self.horizontalLayout.addWidget(self.pushButtonSignUp)
        self.pushButtonGuest = QtWidgets.QPushButton(parent=Form)
        self.pushButtonGuest.setObjectName("pushButtonGuest")
        self.horizontalLayout.addWidget(self.pushButtonGuest)
        self.gridLayout_2.addLayout(self.horizontalLayout, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        # wczytanie danych z pliku zoo.data i zapisanie ich do obiektu DataFrame biblioteki Pandas
        df = pd.read_csv('zoo.data', header=None)
        df_with_labels = pd.DataFrame
        df_with_labels = df.copy()
        selected_indexes = set()
        selected_columns = set()
        selected_rows = set()
        # utworzenie obiektu QTableView
        table_view = QTableView()

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Password:"))
        self.label_2.setText(_translate("Form", "Login:"))
        self.pushButtonSignIn.setText(_translate("Form", "Sign In"))
        self.pushButtonSignUp.setText(_translate("Form", "Sign Up"))
        self.pushButtonGuest.setText(_translate("Form", "Guest"))

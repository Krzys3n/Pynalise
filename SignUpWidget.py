# Form implementation generated from reading ui file 'SignUpWidget.ui'
#
# Created by: PyQt6 UI code generator 6.5.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class SignUpWidget(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(320, 158)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.checkBox = QtWidgets.QCheckBox(parent=Form)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout_2.addWidget(self.checkBox, 0, 1, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEditMail = QtWidgets.QLineEdit(parent=Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEditMail.sizePolicy().hasHeightForWidth())
        self.lineEditMail.setSizePolicy(sizePolicy)
        self.lineEditMail.setObjectName("lineEditMail")
        self.gridLayout.addWidget(self.lineEditMail, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(parent=Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.lineEditLogin = QtWidgets.QLineEdit(parent=Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEditLogin.sizePolicy().hasHeightForWidth())
        self.lineEditLogin.setSizePolicy(sizePolicy)
        self.lineEditLogin.setObjectName("lineEditLogin")
        self.gridLayout.addWidget(self.lineEditLogin, 1, 1, 1, 1)
        self.lineEditPassword = QtWidgets.QLineEdit(parent=Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEditPassword.sizePolicy().hasHeightForWidth())
        self.lineEditPassword.setSizePolicy(sizePolicy)
        self.lineEditPassword.setPlaceholderText("")
        self.lineEditPassword.setClearButtonEnabled(False)
        self.lineEditPassword.setObjectName("lineEditPassword")
        self.gridLayout.addWidget(self.lineEditPassword, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(parent=Form)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.pushButtonSign = QtWidgets.QPushButton(parent=Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButtonSign.sizePolicy().hasHeightForWidth())
        self.pushButtonSign.setSizePolicy(sizePolicy)
        self.pushButtonSign.setObjectName("pushButtonSign")
        self.gridLayout_2.addWidget(self.pushButtonSign, 2, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.checkBox.setText(_translate("Form", "Show Pass"))
        self.label.setText(_translate("Form", "E-mail:"))
        self.label_2.setText(_translate("Form", "Password:"))
        self.label_3.setText(_translate("Form", "Login:"))
        self.pushButtonSign.setText(_translate("Form", "Sign Up"))

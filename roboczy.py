import sys
from PyQt6 import QtWidgets
from PyQt6.QtCore import QSize, Qt

from main_ui import Ui_MainWindow
from LoginWidget import Ui_Form

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

class LoginWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow, *args, obj=None, **kwargs):
        super(LoginWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.mainwindow = mainwindow  # Przechowaj referencję do obiektu MainWindow
        self.pushButtonSignIn.clicked.connect(self.signIn)

    def signIn(self):
        if self.lineEditLogin.text() == "3":
            self.mainwindow.show()  # Pokaż główne okno
            self.hide()  # Ukryj okno logowania
        else:
            print("Niezalogowano")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow()
    login_window = LoginWidget(main_window)  # Przekazanie referencji do obiektu MainWindow
    login_window.show()

    sys.exit(app.exec())

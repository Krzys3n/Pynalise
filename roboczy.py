import sys
from functools import partial

import pandas as pd
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QSize, Qt, QThread, QItemSelectionModel
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QEnterEvent
from PyQt6.QtWidgets import QTableView, QDialog, QLineEdit, QPushButton, QVBoxLayout

import main

from main_ui import Ui_MainWindow
from LoginWidget import LoginWidget
from SignUpWidget import SignUpWidget

from enum import Enum


# Definicja enuma
class Role(Enum):
    ADMIN = 1
    USER = 2
    GUEST = 3


class DataRetrievalThread(QThread):
    def __init__(self, tableView):
        super().__init__()
        self.i = 1
        self.tableView = tableView

    def run(self):
        while True:
            # Wywołanie funkcji retrieveData()
            main.retrieveData(self.tableView)

            print("Wątek "+str(self.i))

            # Zaczekaj przed kolejnym wywołaniem
            self.msleep(1000)  # Czas oczekiwania w milisekundach
            self.i+=1

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    print(main.df.dtypes)
    print(main.df_with_labels.dtypes)


    def search(self, value):
        ilosc_szuk= 0
        # okno dialogowo:

        dialog = QDialog()
        dialog.setWindowTitle("Wpisz szukaną wartość: " )
        dialog.header_field = QLineEdit()
        ok_button = QPushButton("Szukaj")
        ok_button.clicked.connect(dialog.accept)
        anuluj_button = QPushButton("Anuluj")
        anuluj_button.clicked.connect(dialog.reject)  # Przerwanie dialogu po kliknięciu przycisku "Anuluj"
        anuluj_button.setVisible(False)
        layout = QVBoxLayout()
        layout.addWidget(dialog.header_field)
        layout.addWidget(ok_button)
        layout.addWidget(anuluj_button)
        dialog.setLayout(layout)

        if dialog.exec() == 1:
            value = dialog.header_field.text()

        else:
            return 0

        # Clear current selection.
        self.tableView.clearSelection()

        if not value:
            # Empty value, don't search.
            return

        model = self.tableView.model()
        rows = model.rowCount()
        cols = model.columnCount()
        matching_indexes = []

        for row in range(rows):
            for col in range(cols):
                index = model.index(row, col)
                item = model.data(index, Qt.ItemDataRole.DisplayRole)
                if str(item) == value:
                    ilosc_szuk +=1
                    matching_indexes.append(index)
        self.textBrowser.append("Ilość wyszukań wartości " + str(value) + " w tabeli: " + str(ilosc_szuk))
        if matching_indexes:
            # Select all matching indexes.
            selection_model = self.tableView.selectionModel()
            for index in matching_indexes:
                selection_model.select(index, QItemSelectionModel.SelectionFlag.Select)
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        print(LoginWidget.role)

        if (LoginWidget.role == Role.GUEST):
            print("Zalogowany: Gość")
            self.menuMenu.setVisible(False)

            self.actionWczytaj_z_CSV.setVisible(False)
            self.actionZapisz_do_CSV.setVisible(False)
            self.actionGenerateResultsPDF.setVisible(False)
            self.actionWprowadzNagl.setVisible(False)

        # przyciski statystyczne: Obsługa:
        self.pushButtonMinimum.clicked.connect(lambda: main.calculate_minimum(self.textBrowser, self.tableView))
        # self.pushButtonMinimum.clicked.connect(lambda:self.search("1"))
        self.pushButtonMaximum.clicked.connect(lambda: main.calculate_maximum(self.textBrowser, self.tableView))
        self.pushButtonClear.clicked.connect(lambda: self.textBrowser.clear())
        self.pushButtonMean.clicked.connect(lambda: main.calculate_mean(self.textBrowser, self.tableView))
        self.pushButtonStd.clicked.connect(lambda: main.calculate_std(self.textBrowser, self.tableView))
        self.pushButtonMedian.clicked.connect(lambda: main.calculate_median(self.textBrowser, self.tableView))
        self.pushButtonDystrybucja.clicked.connect(lambda: main.generate_distribution_plot(self.tableView))
        self.pushButtonKoorelacja.clicked.connect(
            lambda: main.calculate_coorelation(self.comboBoxAtrybut1, self.comboBoxAtrybut2, self.textBrowser,
                                               self.tableView))
        self.pushButtonCalcChecked.clicked.connect(
            lambda: main.calculate_checked_stats(self.checkBoxMin, self.checkBoxMax, self.checkBoxStd, self.checkBoxMdn,
                                                 self.checkBoxMean, self.textBrowser, self.tableView))
        self.pushButtonHeatmap.clicked.connect(lambda: main.generate_correlation_heatmap(self.tableView))
        self.pushButtonClass.clicked.connect(
            lambda: main.classificate_selected_data(self.comboBoxAtrybutClass, self.textBrowser))
        self.pushButtonPorownaj.clicked.connect(lambda: main.generate_comparison_plot(self.tableView))
        # wczytanie danych z pliku zoo.data i zapisanie ich do obiektu DataFrame biblioteki Pandas
        df = pd.read_csv('zoo.data', header=None)
        df_with_labels = pd.DataFrame
        df_with_labels = df.copy()

        # utworzenie obiektu QTableView
        table_view = QTableView()

        # utworzenie obiektu modelu danych i przypisanie go do tabeli
        model = QStandardItemModel(df.shape[0], df.shape[1])
        model.setHorizontalHeaderLabels([str(i) for i in range(df.shape[1])])
        for row in range(df.shape[0]):
            for column in range(df.shape[1]):
                item = QStandardItem(str(df.iloc[row, column]))
                model.setItem(row, column, item)

        table_view.setModel(model)

        # Ustawianie nazw kolumn
        labels = ["animal name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed",
                  "backbone",
                  "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"]
        labels_no_zero_column = ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed",
                                 "backbone",
                                 "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"]
        model.setHorizontalHeaderLabels(labels)
        df_with_labels.columns = labels

        # Liczba kolumn minus 1
        x = len(df.columns) - 1

        # Ustawienie nazw kolumn na liczby od 0 do x
        df.columns = list(range(x + 1))

        # dodanie tabeli do layoutuu
        table_view = self.tableView
        table_view.setModel(model)

        # Comboboxy pozwalające na wybór atrybutów danych do porównania
        self.comboBoxAtrybut1.addItems(labels_no_zero_column)
        self.comboBoxAtrybut2.addItems(labels_no_zero_column)
        self.comboBoxAtrybutClass.addItems(labels_no_zero_column)

        # wyświetlanie informacji o kolumnie w tabeli
        table_view.setMouseTracking(True)
        table_view.entered.connect(
            partial(main.display_column_info, table_view=self.tableView, textBrowserInfo=self.textBrowserInfo))

        # Wątek odpowiadający za aktualizację danych w DF
        self.thread = DataRetrievalThread(self.tableView)
        self.thread.start()


        # obsługa paska menu
        self.actionCSV_load.triggered.connect(
            lambda: main.load_CSV_file(self.comboBoxAtrybut1, self.comboBoxAtrybut2, self.comboBoxAtrybutClass,
                                       self.tableView))
        self.actionCSV_save.triggered.connect(lambda: main.save_to_XML())
        self.actionJSON_load.triggered.connect(
            lambda: main.load_JSON_file(self.comboBoxAtrybut1, self.comboBoxAtrybut2, self.comboBoxAtrybutClass,
                                       self.tableView))
        self.actionJSON_save.triggered.connect(lambda: main.save_to_JSON())
        self.actionWprowadzNagl.triggered.connect(
            lambda: main.add_headers_manually(self.comboBoxAtrybut1, self.comboBoxAtrybut2,
                                              self.comboBoxAtrybutClass, self.tableView))
        self.actionGenerateResultsPDF.triggered.connect(lambda: main.generate_pdf(self.textBrowser))
        self.actionCiemny.triggered.connect(lambda: main.change_to_darkmode(self, self.tableView, self.textBrowserInfo))
        self.actionJasny.triggered.connect(lambda: main.change_to_lightmode(self, self.tableView))
        self.actionWyszukaj.triggered.connect(lambda: self.search('1'))

        # Obsługa wyświetlania informacji o przyciskach:
        self.pushButtonMinimum.enterEvent = partial(main.on_button_enter,
                                                    button_name=str(self.pushButtonMinimum.objectName()),
                                                    QtTextBrowser=self.textBrowserInfo)
        self.pushButtonMaximum.enterEvent = partial(main.on_button_enter,
                                                    button_name=str(self.pushButtonMaximum.objectName()),
                                                    QtTextBrowser=self.textBrowserInfo)
        self.pushButtonStd.enterEvent = partial(main.on_button_enter,
                                                button_name=str(self.pushButtonStd.objectName()),
                                                QtTextBrowser=self.textBrowserInfo)
        self.pushButtonMedian.enterEvent = partial(main.on_button_enter,
                                                   button_name=str(self.pushButtonMedian.objectName()),
                                                   QtTextBrowser=self.textBrowserInfo)
        self.pushButtonMean.enterEvent = partial(main.on_button_enter,
                                                 button_name=str(self.pushButtonMean.objectName()),
                                                 QtTextBrowser=self.textBrowserInfo)
        self.pushButtonClear.enterEvent = partial(main.on_button_enter,
                                                  button_name=str(self.pushButtonClear.objectName()),
                                                  QtTextBrowser=self.textBrowserInfo)
        self.pushButtonCalcChecked.enterEvent = partial(main.on_button_enter,
                                                        button_name=str(self.pushButtonCalcChecked.objectName()),
                                                        QtTextBrowser=self.textBrowserInfo)
        self.pushButtonKoorelacja.enterEvent = partial(main.on_button_enter,
                                                       button_name=str(self.pushButtonKoorelacja.objectName()),
                                                       QtTextBrowser=self.textBrowserInfo)
        self.pushButtonPorownaj.enterEvent = partial(main.on_button_enter,
                                                     button_name=str(self.pushButtonPorownaj.objectName()),
                                                     QtTextBrowser=self.textBrowserInfo)
        self.pushButtonDystrybucja.enterEvent = partial(main.on_button_enter,
                                                        button_name=str(self.pushButtonDystrybucja.objectName()),
                                                        QtTextBrowser=self.textBrowserInfo)
        self.pushButtonHeatmap.enterEvent = partial(main.on_button_enter,
                                                    button_name=str(self.pushButtonHeatmap.objectName()),
                                                    QtTextBrowser=self.textBrowserInfo)
        self.pushButtonClass.enterEvent = partial(main.on_button_enter,
                                                  button_name=str(self.pushButtonClass.objectName()),
                                                  QtTextBrowser=self.textBrowserInfo)

        # Wyświetlanie informacji na temat przycisków - pobieranie danych na temat zaznaczonych kolumn/wierszy/indeksów:
        table_view.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.selection_model = table_view.selectionModel()
        self.selection_model.selectionChanged.connect(lambda: main.handle_selection_changed(table_view))


class LoginWidget(QtWidgets.QWidget, LoginWidget):
    role = Role.GUEST

    def __init__(self, sign_up_window, *args, obj=None, **kwargs):
        super(LoginWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.sign_up_window = sign_up_window
        self.pushButtonSignIn.clicked.connect(self.signIn)
        self.pushButtonSignUp.clicked.connect(self.signUp)
        self.pushButtonGuest.clicked.connect(self.guestLogIn)

    def signIn(self, main_window):
        if self.lineEditLogin.text() == "3":
            LoginWidget.role = Role.ADMIN
        else:
            LoginWidget.role = Role.GUEST

        main_window = MainWindow()
        main_window.show()  # Pokaż główne okno
        self.hide()  # Ukryj okno logowania
        print(LoginWidget.role)

    def signUp(self):
        self.sign_up_window.show()  # Pokaż główne okno
        self.hide()  # Ukryj okno logowania

    def guestLogIn(self):
        self.hide()
        main_window = MainWindow()
        main_window.show()  # Pokaż główne okno


class SignUpWidget(QtWidgets.QWidget, SignUpWidget):
    @staticmethod
    def setLoginWindow(login_window):
        LoginWidget.sign_up_window = login_window

    def __init__(self, *args, obj=None, **kwargs):
        super(SignUpWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.pushButtonSign.clicked.connect(self.signUp)

    def signUp(self):
        print("Zarejestrowano")
        login_window.show()
        self.hide()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    sign_up_window = SignUpWidget()
    login_window = LoginWidget(sign_up_window)
    login_window.show()

    sys.exit(app.exec())

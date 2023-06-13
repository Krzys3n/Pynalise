import os
import sys
from functools import partial
import numpy as np
import seaborn as sns
from PyQt6 import uic, QtGui, QtWidgets, QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QApplication, QTableView, QFileDialog, QMessageBox, QDialog, QPushButton, QVBoxLayout, \
    QLineEdit, QHBoxLayout, QStyleFactory
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtGui import QEnterEvent

# obliczenia statystyczne:

def calculate_minimum(text_browser):
    try:
        for column in selected_columns:
            # Oblicz minimum z wybranej kolumny
            minimum = df[column].min()

            # Konwertuj wartość na string
            minimum_str = str(minimum)

            # Pobierz tekst nagłówka kolumny z QTableView
            header_text = table_view.model().headerData(column, Qt.Orientation.Horizontal)

            # Wyświetl wartość w QTextBrowser
            text_browser.append(f"Minimum from column {header_text} is: {minimum_str}")
    except Exception as e:
        text_browser.append(f"Error occurred: {str(e)}")


def calculate_maximum(text_browser):
    try:
        for column in selected_columns:
            # Oblicz maksimum z wybranej kolumny
            maximum = df[column].max()

            # Konwertuj wartość na string
            maximum_str = str(maximum)

            # Pobierz tekst nagłówka kolumny z QTableView
            header_text = table_view.model().headerData(column, Qt.Orientation.Horizontal)

            # Wyświetl wartość w QTextBrowser
            text_browser.append(f"Maximum from column {header_text} is: {maximum_str}")
    except Exception as e:
        text_browser.append(f"Error occurred: {str(e)}")


def calculate_median(text_browser):
    try:
        for column in selected_columns:
            # Oblicz medianę z wybranej kolumny
            median = df[column].median()

            # Konwertuj wartość na string
            median_str = str(median)

            # Pobierz tekst nagłówka kolumny z QTableView
            header_text = table_view.model().headerData(column, Qt.Orientation.Horizontal)

            # Wyświetl wartość w QTextBrowser
            text_browser.append(f"Median from column {header_text} is: {median_str}")
    except Exception as e:
        text_browser.append(f"Error occurred: {str(e)}")


def calculate_std(text_browser):
    try:
        for column in selected_columns:
            # Oblicz odchylenie standardowe z wybranej kolumny
            std = df[column].std()

            # Konwertuj wartość na string
            std_str = str(std)

            # Pobierz tekst nagłówka kolumny z QTableView
            header_text = table_view.model().headerData(column, Qt.Orientation.Horizontal)

            # Wyświetl wartość w QTextBrowser
            text_browser.append(f"Standard deviation from column {header_text} is: {std_str}")
    except Exception as e:
        text_browser.append(f"Error occurred: {str(e)}")


def calculate_mean(text_browser):
    try:
        for column in selected_columns:
            # Oblicz średnią z wybranej kolumny
            mean = df[column].mean()

            # Konwertuj wartość na string
            mean_str = str(mean)

            # Pobierz tekst nagłówka kolumny z QTableView
            header_text = table_view.model().headerData(column, Qt.Orientation.Horizontal)

            # Wyświetl wartość w QTextBrowser
            text_browser.append(f"Mean from column {header_text} is: {mean_str}")
    except Exception as e:
        text_browser.append(f"Error occurred: {str(e)}")


def calculate_coorelation():
    model = table_view.model()
    # pobierz indeks wybranej kolumny z comboBoxAtrybut1
    column1_index = form.comboBoxAtrybut1.currentIndex() + 1
    # pobierz indeks wybranej kolumny z comboBoxAtrybut2
    column2_index = form.comboBoxAtrybut2.currentIndex() + 1
    # pobierz dane z wybranej kolumny 1
    column1_data = [model.data(model.index(row, column1_index)) for row in range(model.rowCount())]
    # pobierz dane z wybranej kolumny 2
    column2_data = [model.data(model.index(row, column2_index)) for row in range(model.rowCount())]
    # utwórz nowy obiekt DataFrame z pobranych danych
    data = pd.DataFrame(
        {form.comboBoxAtrybut1.currentText(): column1_data, form.comboBoxAtrybut2.currentText(): column2_data})
    data = data.apply(pd.to_numeric)
    correlation_matrix = data[[form.comboBoxAtrybut1.currentText(), form.comboBoxAtrybut2.currentText()]].corr()
    correlation_coefficient = correlation_matrix.iloc[0, 1]
    form.textBrowser.append("Koorelacja między tymi atrybutami wynosi: " + str(correlation_coefficient))





# Podłączanie GUI poprzez plik ui stworzony QT designerze

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'main.ui')
Form, Window = uic.loadUiType(filename)

app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.setFixedSize(1060, 760)


# wczytanie danych z pliku zoo.data i zapisanie ich do obiektu DataFrame biblioteki Pandas
df = pd.read_csv('zoo.data', header=None)
df_with_labels = pd.DataFrame
df_with_labels = df.copy()
selected_indexes = set()
selected_columns = set()
selected_rows = set()
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
labels = ["animal name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone",
          "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"]
labels_no_zero_column = ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone",
                         "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"]
model.setHorizontalHeaderLabels(labels)
df_with_labels.columns = labels

# Liczba kolumn minus 1
x = len(df.columns) - 1

# Ustawienie nazw kolumn na liczby od 0 do x
df.columns = list(range(x + 1))

# dodanie tabeli do layoutuu
table_view = form.tableView
table_view.setModel(model)

# przyciski:
form.pushButtonMinimum.clicked.connect(lambda: calculate_minimum(form.textBrowser))
form.pushButtonMaximum.clicked.connect(lambda: calculate_maximum(form.textBrowser))
form.pushButtonClear.clicked.connect(lambda: form.textBrowser.clear())
form.pushButtonMean.clicked.connect(lambda: calculate_mean(form.textBrowser))
form.pushButtonStd.clicked.connect(lambda: calculate_std(form.textBrowser))
form.pushButtonMedian.clicked.connect(lambda: calculate_median(form.textBrowser))
form.pushButtonDystrybucja.clicked.connect(lambda: generate_distribution_plot())
form.pushButtonKoorelacja.clicked.connect(lambda: calculate_coorelation())
form.pushButtonCalcChecked.clicked.connect(lambda: calculate_checked_stats())
form.pushButtonHeatmap.clicked.connect(lambda: generate_correlation_heatmap())

class CustomHeaderView(QtWidgets.QHeaderView):
    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        painter.fillRect(rect, QtGui.QColor("black"))
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, table_view.model().headerData(logicalIndex, QtCore.Qt.Orientation.Horizontal))
        painter.restore()
class CustomHeaderView2(QtWidgets.QHeaderView):
    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        painter.fillRect(rect, QtGui.QColor("black"))
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(table_view.model().headerData(logicalIndex, QtCore.Qt.Orientation.Vertical)))
        painter.restore()
class CustomHeaderView3(QtWidgets.QHeaderView):
    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        painter.fillRect(rect, QtGui.QColor("white"))
        painter.setPen(QtGui.QColor("black"))
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, table_view.model().headerData(logicalIndex, QtCore.Qt.Orientation.Horizontal))
        painter.restore()
class CustomHeaderView4(QtWidgets.QHeaderView):
    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        painter.fillRect(rect, QtGui.QColor("white"))
        painter.setPen(QtGui.QColor("black"))
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(table_view.model().headerData(logicalIndex, QtCore.Qt.Orientation.Vertical)))
        painter.restore()



def change_to_darkmode():
    window.setStyleSheet("""
            background-color: #262626;
            color: white;
            font-family: Titillium;
            font-size: 14px;

        }
            """)
    table_view.setStyleSheet("""
            background-color: #262626;
            color: white;
            font-family: Titillium;
            font-size: 18px;

        }
            """)

    hor_header = CustomHeaderView(QtCore.Qt.Orientation.Horizontal)
    ver_header = CustomHeaderView2(QtCore.Qt.Orientation.Vertical)
    table_view.setHorizontalHeader(hor_header)
    table_view.setVerticalHeader(ver_header)
    model = table_view.model()

    for row in range(model.rowCount()):
        for column in range(model.columnCount()):
            item = model.item(row, column)
            item.setForeground(QtGui.QColor("white"))

    table_view.horizontalHeader().setVisible(True)
    table_view.verticalHeader().setVisible(True)
    form.textBrowserInfo.setStyleSheet("font-size: 24px;")

def change_to_lightmode():
    window.setStyleSheet("")
    hor_header = CustomHeaderView3(QtCore.Qt.Orientation.Horizontal)
    ver_header = CustomHeaderView4(QtCore.Qt.Orientation.Vertical)
    table_view.setHorizontalHeader(hor_header)
    table_view.setVerticalHeader(ver_header)
    table_view.horizontalHeader().setVisible(True)
    table_view.verticalHeader().setVisible(True)
    table_view.setStyleSheet("""
            background-color: #fffff;
            color: black;
            font-family: Titillium;
            font-size: 18px;

        }
            """)
    model = table_view.model()

    for row in range(model.rowCount()):
        for column in range(model.columnCount()):
            item = model.item(row, column)
            item.setForeground(QtGui.QColor("black"))





# wyświetlanie informacji o danej kolumnie

def switch_dictionary(column_name):
    switcher = {
        "animal name": "The name of the animal",
        "hair": "Whether the animal has hair or not",
        "feathers": "Whether the animal has feathers or not",
        "eggs": "Whether the animal lays eggs or not",
        "milk": "Whether the animal produces milk or not",
        "airborne": "Whether the animal can fly or not",
        "aquatic": "Whether the animal lives in water or not",
        "predator": "Whether the animal is a predator or not",
        "toothed": "Whether the animal has teeth or not",
        "backbone": "Whether the animal has a backbone or not",
        "breathes": "Whether the animal breathes air or not",
        "venomous": "Whether the animal is venomous or not",
        "fins": "Whether the animal has fins or not",
        "legs": "Number of legs that the animal has",
        "tail": "Whether the animal has a tail or not",
        "domestic": "Whether the animal is domesticated or not",
        "catsize": "Whether the animal is cat-sized or not",
        "type": "Type of animal (mammal, bird, reptile, etc.)"
    }
    return switcher.get(column_name, "No info about Column")


def display_column_info(index):
    column_name = table_view.model().headerData(index.column(), Qt.Orientation.Horizontal)
    form.textBrowserInfo.setText(switch_dictionary(column_name))
    form.textBrowserInfo.setAlignment(Qt.AlignmentFlag.AlignCenter)



table_view.setMouseTracking(True)
table_view.entered.connect(display_column_info)

# Comboboxy pozwalające na wybór atrybutów danych do porównania

form.comboBoxAtrybut1.addItems(labels_no_zero_column)
form.comboBoxAtrybut2.addItems(labels_no_zero_column)

# Generowanie wykresów:
def generate_comparison_plot():
    model = form.tableView.model()

    data = pd.DataFrame()  # Utwórz pusty obiekt DataFrame

    for column_index in selected_columns:
        # Pobierz nazwę wybranej kolumny z QTableView
        column_name = model.headerData(column_index, Qt.Orientation.Horizontal)

        # Pobierz dane z wybranej kolumny
        column_data = [model.data(model.index(row, column_index)) for row in range(model.rowCount())]

        # Dodaj dane do obiektu DataFrame
        data[column_name] = column_data

    # Konwertuj dane na typ numeryczny
    data = data.apply(pd.to_numeric, errors='coerce')
    counts = data.apply(pd.Series.value_counts).fillna(0)
    counts.plot(kind='bar')

    plt.xlabel('Atrybut')
    plt.ylabel('Suma')

    # Skonstruuj tytuł wykresu na podstawie nazw zaznaczonych atrybutów
    title = "Zliczenia "
    for column_name in data.columns:
        title += str(column_name) + " + "

    title = title[:-3]  # Usuń ostatni znak "+" i spacje

    plt.title(title)

    plt.show()


form.pushButtonPorownaj.clicked.connect(generate_comparison_plot)

form.comboBoxAtrybut1.setCurrentIndex(3)
form.comboBoxAtrybut2.setCurrentIndex(0)



##########################

##Obsługiwanie paska menu
def wczytaj_plik_csv():
    global table_view
    global df
    global df_with_labels
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Wczytywanie pliku CSV")
    msg_box.setText("Czy chcesz wczytać plik z nazwami kolumn?")

    yes_button = msg_box.addButton(QMessageBox.StandardButton.Yes)
    yes_button.setText("Tak, z nazwami")
    msg_box.addButton(QMessageBox.StandardButton.No).setText("Nie, bez nazw")
    cn_button = msg_box.addButton(QMessageBox.StandardButton.Cancel)
    cn_button.setVisible(False)

    msg_box.setDefaultButton(yes_button)

    reply = msg_box.exec()

    if reply == 16384:
        header = 0
    elif reply == 65536:
        header = None
    else:
        return

    current_dir = os.getcwd()
    filename, _ = QFileDialog.getOpenFileName(None, "Wybierz plik CSV lub DATA", current_dir,
                                              "All files (*);;Pliki CSV (*.csv);;Pliki DATA (*.data)")
    if not filename:
        # Wyjście z funkcji, jeśli nie został wybrany plik
        return

    # Wczytaj dane z pliku CSV do obiektu DataFrame z biblioteki Pandas
    df = pd.read_csv(filename, header=header)
    df_with_labels = df.copy()
    # Wyświetl dane za pomocą funkcji print

    model = QStandardItemModel(df.shape[0], df.shape[1])
    headers = list(df.columns)
    if reply == 16384:
        model.setHorizontalHeaderLabels(headers)

    for row in range(df.shape[0]):
        for column in range(df.shape[1]):
            item = QStandardItem(str(df.iloc[row, column]))
            model.setItem(row, column, item)

    table_view.setModel(model)

    # wczytywanie nazw kolumn do comboboxów
    column_names = df.columns.tolist()[1:]
    form.comboBoxAtrybut1.clear()
    form.comboBoxAtrybut2.clear()
    form.comboBoxAtrybut1.addItems([str(x) for x in column_names])
    form.comboBoxAtrybut2.addItems([str(x) for x in column_names])

    # Liczba kolumn minus 1
    x = len(df.columns) - 1

    # Ustawienie nazw kolumn na liczby od 0 do x
    df.columns = list(range(x + 1))

    table_view.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
    form.selection_model = table_view.selectionModel()
    form.selection_model.selectionChanged.connect(handle_selection_changed)


def zapisz_plikCSV():
    # Okno dialogowe z pytaniem o zapisanie z nazwami kolumn lub bez
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Zapisywanie pliku CSV")
    msg_box.setText("Czy chcesz zapisać plik z nazwami kolumn?")

    yes_button = msg_box.addButton(QMessageBox.StandardButton.Yes)
    yes_button.setText("Tak")
    msg_box.addButton(QMessageBox.StandardButton.No).setText("Nie")
    cn_button = msg_box.addButton(QMessageBox.StandardButton.Cancel)
    cn_button.setVisible(False)

    msg_box.setDefaultButton(yes_button)

    reply = msg_box.exec()

    if reply == 16384:
        header = True
    elif reply == 65536:
        header = False
    else:
        return

    current_dir = os.getcwd()
    filename, _ = QFileDialog.getSaveFileName(None, "Zapisz plik CSV", current_dir, "Pliki CSV (*.csv)")

    if filename:
        df_with_labels.to_csv(filename, index=False, header=header)


def reczne_wpisywanie_naglowkow():
    global labels
    global table_view
    global df
    global df_with_labels
    labels = []
    for i in df.columns:
        dialog = QDialog()
        dialog.setWindowTitle("Nazwa nagłówka " + str(i))

        dialog.header_field = QLineEdit()

        ok_button = QPushButton("OK")
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
            label = dialog.header_field.text()
            labels.append(label)
        else:
            return 0
    model = QStandardItemModel(df.shape[0], df.shape[1])
    model.setHorizontalHeaderLabels(labels)

    for row in range(df.shape[0]):
        for column in range(df.shape[1]):
            item = QStandardItem(str(df.iloc[row, column]))
            model.setItem(row, column, item)

    table_view.setModel(model)

    df_with_labels.columns = labels

    # wczytywanie nazw kolumn do comboboxów
    form.comboBoxAtrybut1.clear()
    form.comboBoxAtrybut2.clear()
    form.comboBoxAtrybut1.addItems([str(x) for x in labels])
    form.comboBoxAtrybut2.addItems([str(x) for x in labels])

    print(df_with_labels.columns)


def funkcja():
    print(labels)


def generate_correlation_heatmap():
    model = form.tableView.model()

    data = pd.DataFrame()  # Utwórz pusty obiekt DataFrame

    for column_index in selected_columns:
        # Pobierz nazwę wybranej kolumny z QTableView
        column_name = model.headerData(column_index, Qt.Orientation.Horizontal)

        # Pobierz dane z wybranej kolumny
        column_data = [model.data(model.index(row, column_index)) for row in range(model.rowCount())]

        # Dodaj dane do obiektu DataFrame
        data[column_name] = column_data

    # Konwertuj dane na typ numeryczny
    data = data.apply(pd.to_numeric, errors='coerce')

    # Oblicz macierz korelacji
    corr_matrix = data.corr()

    # Wygeneruj heatmapę korelacji
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap - Korelacja')
    plt.show()



def generate_distribution_plot():
    model = form.tableView.model()

    data = pd.DataFrame()  # Utwórz pusty obiekt DataFrame

    for column_index in selected_columns:
        # Pobierz nazwę wybranej kolumny z QTableView
        column_name = model.headerData(column_index, Qt.Orientation.Horizontal)

        # Pobierz dane z wybranej kolumny
        column_data = [model.data(model.index(row, column_index)) for row in range(model.rowCount())]

        # Dodaj dane do obiektu DataFrame
        data[column_name] = column_data

    # Konwertuj dane na typ numeryczny
    data = data.apply(pd.to_numeric , errors='coerce')

    # Sumuj wartości dla każdej kolumny
    counts = data.sum()

    # Wygeneruj wykres kołowy
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    plt.axis('equal')

    # Skonstruuj tytuł wykresu na podstawie nazw zaznaczonych atrybutów
    title = "Rozkład "
    for column_name in data.columns:
        title += str(column_name) + " + "

    title = title[:-3]  # Usuń ostatni znak "+" i spacje

    plt.title(title)

    plt.show()


def calculate_checked_stats():
    if (form.checkBoxMin.isChecked() == True):
        calculate_minimum(form.textBrowser)
    if (form.checkBoxMax.isChecked() == True):
        calculate_maximum(form.textBrowser)
    if (form.checkBoxStd.isChecked() == True):
        calculate_std(form.textBrowser)
    if (form.checkBoxMdn.isChecked() == True):
        calculate_median(form.textBrowser)
    if (form.checkBoxMean.isChecked() == True):
        calculate_mean(form.textBrowser)


# obsługa zaznaczania rzeczy w tabeli
def handle_selection_changed():
    global selected_columns
    global selected_rows
    global table_view
    selected_indexes = table_view.selectionModel().selectedIndexes()
    selected_columns.clear()
    selected_rows.clear()

    # Sprawdź, czy liczba kolumn się zmieniła
    model = table_view.model()
    if model.columnCount() != len(selected_columns):
        # Zresetuj listę selected_columns
        selected_columns = set()

    for index in selected_indexes:
        selected_columns.add(index.column())
        selected_rows.add(index.row())
        print(index.column())


table_view.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
form.selection_model = table_view.selectionModel()
form.selection_model.selectionChanged.connect(handle_selection_changed)


def generate_pdf():
    # Wyświetl okno dialogowe do wyboru miejsca zapisu pliku
    file_dialog = QFileDialog()
    file_dialog.setWindowTitle("Save PDF")
    file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
    file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    file_dialog.setNameFilter("PDF Files (*.pdf)")
    file_dialog.setDefaultSuffix("pdf")

    if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
        file_path = file_dialog.selectedFiles()[0]
    else:
        return  # Użytkownik anulował wybór pliku

    if file_path:
        # Pobierz tekst z QTextBrowser
        text = form.textBrowser.toPlainText()

        # Utwórz nowy obiekt Canvas
        c = canvas.Canvas(file_path)

        # Ustaw czcionkę i rozmiar tekstu
        c.setFont("Helvetica", 12)

        # Wypisz tekst w pliku PDF
        lines = text.split("\n")
        y = 800  # Wysokość początkowa
        for line in lines:
            c.drawString(50, y, line)
            y -= 20  # Zmniejsz wysokość dla kolejnej linii

        # Zamknij plik PDF
        c.save()


# obsługa paska menu:
form.actionWczytaj_z_CSV.triggered.connect(wczytaj_plik_csv)
form.actionZapisz_do_CSV.triggered.connect(zapisz_plikCSV)
form.actionWprowadzNagl.triggered.connect(reczne_wpisywanie_naglowkow)
form.actionGenerateResultsPDF.triggered.connect(generate_pdf)
form.actionCiemny.triggered.connect(change_to_darkmode)
form.actionJasny.triggered.connect(change_to_lightmode)

# wyświetlanie informacji o przyciskach:
def switch_dictionary_buttons(button_name):
    switcher = {
        "pushButtonMinimum": "Calculates the minimum of selected columns data",
        "pushButtonMaximum": "Calculates the maximum of selected columns data",
        "pushButtonStd": "Calculates the standard deviation of selected columns data",
        "pushButtonMedian": "Calculates the median of selected columns data",
        "pushButtonMean": "Calculates the arithmetic average of selected columns data",
        "pushButtonClear": "Clears the result window",
        "pushButtonCalcChecked": "Designates selected descriptive statistics",
        "pushButtonKoorelacja": "Calculates the correlation of selected attributes and generates a correlation heatmap",
        "pushButtonPorownaj": "Generates a comparison plot of selected columns data",
        "pushButtonDystrybucja": "Generates a distribution plot of selected columns data",
        "pushButtonHeatmap": "Generates a coorelation Heatmap of selected columns data"
    }
    return switcher.get(button_name, "No information available for the button")


def on_button_enter(event: QEnterEvent, button_name: str):
    info = switch_dictionary_buttons(button_name)
    form.textBrowserInfo.setText(info)
    form.textBrowserInfo.setAlignment(Qt.AlignmentFlag.AlignCenter)


# Obsługa wyświetlania informacji o przyciskach:
form.pushButtonMinimum.enterEvent = partial(on_button_enter, button_name=str(form.pushButtonMinimum.objectName()))
form.pushButtonMaximum.enterEvent = partial(on_button_enter, button_name=str(form.pushButtonMaximum.objectName()))
form.pushButtonStd.enterEvent = partial(on_button_enter, button_name=str(form.pushButtonStd.objectName()))
form.pushButtonMedian.enterEvent = partial(on_button_enter, button_name=str(form.pushButtonMedian.objectName()))
form.pushButtonMean.enterEvent = partial(on_button_enter, button_name=str(form.pushButtonMean.objectName()))
form.pushButtonClear.enterEvent = partial(on_button_enter, button_name=str(form.pushButtonClear.objectName()))
form.pushButtonCalcChecked.enterEvent = partial(on_button_enter,
                                                button_name=str(form.pushButtonCalcChecked.objectName()))
form.pushButtonKoorelacja.enterEvent = partial(on_button_enter, button_name=str(form.pushButtonKoorelacja.objectName()))
form.pushButtonPorownaj.enterEvent = partial(on_button_enter, button_name=str(form.pushButtonPorownaj.objectName()))
form.pushButtonDystrybucja.enterEvent = partial(on_button_enter,
                                                button_name=str(form.pushButtonDystrybucja.objectName()))
form.pushButtonHeatmap.enterEvent = partial(on_button_enter,
                                                button_name=str(form.pushButtonHeatmap.objectName()))

# włączanie okna aplikacji:
window.show()
app.exec()

import os
import sys

import numpy as np
import seaborn as sns
from PyQt5.uic.properties import QtCore
from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QApplication, QTableView, QFileDialog, QMessageBox
import pandas as pd
import matplotlib.pyplot as plt


# def calculate_minimum(df, column, text_browser):
#     try:
#         # Oblicz minimum z wybranej kolumny
#         minimum = df[column].min()
#
#         # Konwertuj wartość na string
#         minimum_str = str(minimum)
#
#         # Pobierz tekst nagłówka kolumny z QTableView
#         header_text ="" #table_view.model().headerData(column, Qt.Orientation.Horizontal)
#
#         # Wyświetl wartość w QTextBrowser
#         text_browser.append(f"Minimum from column {header_text} is: {minimum_str}")
#     except Exception as e:
#         text_browser.append(f"Error occurred: {str(e)}")

def calculate_minimum(column, text_browser):
    try:


        # Oblicz minimum z wybranej kolumny
        minimum = df[column].min()

        # Konwertuj wartość na string
        minimum_str = str(minimum)

        # Pobierz tekst nagłówka kolumny z QTableView
        header_text =table_view.model().headerData(column, Qt.Orientation.Horizontal)

        # Wyświetl wartość w QTextBrowser
        text_browser.append(f"Minimum from column {header_text} is: {minimum_str}")
    except Exception as e:
        text_browser.append(f"Error occurred: {str(e)}")

def calculate_maximum(df, column, text_browser):
    try:
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



def calculate_median(df, column, text_browser):
    try:
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


def calculate_std(df, column, text_browser):
    try:
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


def calculate_mean(df, column, text_browser):
    try:
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
labels_no_zero_column = [ "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone",
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
form.pushButtonMinimum.clicked.connect(lambda: calculate_minimum(table_view.currentIndex().column(), form.textBrowser))
form.pushButtonMaximum.clicked.connect(lambda: calculate_maximum(df, table_view.currentIndex().column(), form.textBrowser))
form.pushButtonClear.clicked.connect(lambda: form.textBrowser.clear())
form.pushButtonMean.clicked.connect(lambda: calculate_mean(df, table_view.currentIndex().column(), form.textBrowser))
form.pushButtonStd.clicked.connect(lambda: calculate_std(df, table_view.currentIndex().column(), form.textBrowser))
form.pushButtonMedian.clicked.connect(lambda: calculate_median(df, table_view.currentIndex().column(), form.textBrowser))

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

table_view.setMouseTracking(True)
table_view.entered.connect(display_column_info)


# Comboboxy pozwalające na wybór atrybutów danych do porównania

form.comboBoxAtrybut1.addItems(labels_no_zero_column)
form.comboBoxAtrybut2.addItems(labels_no_zero_column)

# Tworzenie wykresu

def generate_plot(attribute_1, attribute_2):
    model = form.tableView.model()
    # pobierz indeks wybranej kolumny z comboBoxAtrybut1
    column1_index = form.comboBoxAtrybut1.currentIndex()+1
    # pobierz indeks wybranej kolumny z comboBoxAtrybut2
    column2_index = form.comboBoxAtrybut2.currentIndex()+1
    # pobierz dane z wybranej kolumny 1
    column1_data = [model.data(model.index(row, column1_index)) for row in range(model.rowCount())]
    # pobierz dane z wybranej kolumny 2
    column2_data = [model.data(model.index(row, column2_index)) for row in range(model.rowCount())]
    # utwórz nowy obiekt DataFrame z pobranych danych
    data = pd.DataFrame(
        {form.comboBoxAtrybut1.currentText(): column1_data, form.comboBoxAtrybut2.currentText(): column2_data})
    data = data.apply(pd.to_numeric)
    counts = data.apply(pd.Series.value_counts).fillna(0)
    counts.plot(kind='bar')

    plt.xlabel('Atrybut')
    plt.ylabel('Suma')
    text1 = attribute_2
    text2 = attribute_1
    if attribute_1 != attribute_2:
        plt.title('Porównanie sumy '+attribute_1+' oraz '+attribute_2)
    else:
        plt.title('Suma ' + attribute_1)
    plt.show()



form.pushButtonPorownaj.clicked.connect(lambda:generate_plot(form.comboBoxAtrybut1.currentText(), form.comboBoxAtrybut2.currentText() ))

form.comboBoxAtrybut1.setCurrentIndex(3)
form.comboBoxAtrybut2.setCurrentIndex(0)

def calculate_coorelation():
    model = table_view.model()
    # pobierz indeks wybranej kolumny z comboBoxAtrybut1
    column1_index = form.comboBoxAtrybut1.currentIndex()+1
    # pobierz indeks wybranej kolumny z comboBoxAtrybut2
    column2_index = form.comboBoxAtrybut2.currentIndex()+1
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

    # Create a heatmap
    sns.heatmap(correlation_matrix)

    # Add title
    plt.title('Correlation Heatmap')

    # Display the plot
    plt.show()


form.pushButtonKoorelacja.clicked.connect(lambda:calculate_coorelation() )
##########################

##Obsługiwanie paska menu
def wczytaj_plik_csv():
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

    filename, _ = QFileDialog.getOpenFileName(None, "Wybierz plik CSV lub DATA", "",
                                              "All files (*);;Pliki CSV (*.csv);;Pliki DATA (*.data)")

    if not filename:
        # Wyjście z funkcji, jeśli nie został wybrany plik
        return

    global table_view
    global df




    # Wczytaj dane z pliku CSV do obiektu DataFrame z biblioteki Pandas
    df = pd.read_csv(filename, header = header)
    print(df)
    df_with_labels = df.copy()
    # Wyświetl dane za pomocą funkcji print


    model = QStandardItemModel(df.shape[0], df.shape[1])
    headers = list(df.columns)
    print(headers)
    if reply == 16384:
        model.setHorizontalHeaderLabels(headers)



    for row in range(df.shape[0]):
        for column in range(df.shape[1]):
            item = QStandardItem(str(df.iloc[row, column]))
            model.setItem(row, column, item)

    table_view.setModel(model)

    #wczytywanie nazw kolumn do comboboxów
    column_names = df.columns.tolist()[1:]
    form.comboBoxAtrybut1.clear()
    form.comboBoxAtrybut2.clear()
    form.comboBoxAtrybut1.addItems([str(x) for x in column_names])
    form.comboBoxAtrybut2.addItems([str(x) for x in column_names])


    # Liczba kolumn minus 1
    x = len(df.columns) - 1

    # Ustawienie nazw kolumn na liczby od 0 do x
    df.columns = list(range(x + 1))


def zapisz_plikCSV():
    # Okno dialogowe z pytaniem o zapisanie z nazwami kolumn lub bez
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Zapisywanie pliku CSV")
    msg_box.setText("Czy chcesz zapisać plik z nazwami kolumn?")

    yes_button = msg_box.addButton(QMessageBox.StandardButton.Yes)
    yes_button.setText("Tak")
    msg_box.addButton(QMessageBox.StandardButton.No).setText("Nie")
    cn_button=  msg_box.addButton(QMessageBox.StandardButton.Cancel)
    cn_button.setVisible(False)

    msg_box.setDefaultButton(yes_button)

    reply = msg_box.exec()

    if reply == 16384:
        header = True
    elif reply == 65536:
        header = False
    else:
        return


    filename, _ = QFileDialog.getSaveFileName(None, "Zapisz plik CSV", "", "Pliki CSV (*.csv)")

    if filename:
        df_with_labels.to_csv(filename, index=False, header=header)

def funkcja():
    distribution()


form.actionWczytaj_z_CSV.triggered.connect(wczytaj_plik_csv)
form.actionZapisz_do_CSV.triggered.connect(zapisz_plikCSV)
form.actionfunkcja.triggered.connect(funkcja)


def distribution():
    counts = df[table_view.currentIndex().column()].value_counts()
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Distribution of ' + df_with_labels.columns[table_view.currentIndex().column()])
    plt.show()



# włączanie okna aplikacji
window.show()
app.exec()




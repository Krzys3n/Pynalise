from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QApplication, QTableView
import pandas as pd





def calculate_minimum(df, column, text_browser):
    try:
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

Form, Window = uic.loadUiType("C:/Users/Krzyś/PycharmProjects/Zoo_Project/main.ui")

app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)


# wczytanie danych z pliku zoo.data i zapisanie ich do obiektu DataFrame biblioteki Pandas
df = pd.read_csv('zoo.data', header=None)

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
model.setHorizontalHeaderLabels(["animal name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone",
          "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"])


# dodanie tabeli do layoutuu
table_view = form.tableView
table_view.setModel(model)


# przyciski:
form.pushButtonMinimum.clicked.connect(lambda: calculate_minimum(df, table_view.currentIndex().column(), form.textBrowser))
form.pushButtonMaximum.clicked.connect(lambda: calculate_maximum(df, table_view.currentIndex().column(), form.textBrowser))
form.pushButtonClear.clicked.connect(lambda: form.textBrowser.clear())
form.pushButtonMean.clicked.connect(lambda: calculate_mean(df, table_view.currentIndex().column(), form.textBrowser))
form.pushButtonStd.clicked.connect(lambda: calculate_std(df, table_view.currentIndex().column(), form.textBrowser))
form.pushButtonMedian.clicked.connect(lambda: calculate_median(df, table_view.currentIndex().column(), form.textBrowser))

column_text = []
header = table_view.horizontalHeader()


window.show()
app.exec()




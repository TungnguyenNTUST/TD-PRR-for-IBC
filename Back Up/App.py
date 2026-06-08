# #-----------------------------------------------------
# from PySide6.QtWidgets import QApplication, QWidget
#
# # Only needed for access to command line arguments
# import sys
#
# # You need one (and only one) QApplication instance per application.
# # Pass in sys.argv to allow command line arguments for your app.
# # If you know you won't use command line arguments QApplication([]) works too.
# app = QApplication(sys.argv)
#
# # Create a Qt widget, which will be our window.
# window = QWidget()
# window.show()  # IMPORTANT!!!!! Windows are hidden by default.
#
# # Start the event loop.
# app.exec()
#
# # Your application won't reach here until you exit and the event
# # loop has stopped.
#

#-----------------------------------------------------
# import sys
# from PySide6.QtWidgets import QApplication, QPushButton
# #
# app = QApplication(sys.argv)
#
# window = QPushButton("Push Me")
# window.show()
#
# app.exec()


# #-----------------------------------------------------
# import sys
# from PySide6.QtWidgets import QApplication, QMainWindow
#
# app = QApplication(sys.argv)
#
# window = QMainWindow()
# window.show()
#
# # Start the event loop.
# app.exec()

# #-----------------------------------------------------
# import sys
#
# from PySide6.QtCore import QSize, Qt
# from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
#
#
# # Subclass QMainWindow to customize your application's main window
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("PEDAT-Accelerating Your Future Design ")
#
#         button = QPushButton("Press Me!")
#
#         # Set the central widget of the Window.
#         self.setCentralWidget(button)
#
#
# app = QApplication(sys.argv)
#
# window = MainWindow()
# window.show()
#
# app.exec()


# #-----------------------------------------------------
# import sys
#
# from PySide6.QtCore import QSize, Qt
# from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
#
#
# # Subclass QMainWindow to customize your application's main window
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("PEDAT-Accelerating Your Future Design")
#
#         button = QPushButton("Press Me!")
#
#         self.setFixedSize(QSize(800, 600))
#
#         # Set the central widget of the Window.
#         self.setCentralWidget(button)
#
#
# app = QApplication(sys.argv)
#
# window = MainWindow()
# window.show()
#
# app.exec()

# #-----------------------------------------------------
# import sys
# from PySide6.QtWidgets import QApplication, QMainWindow
#
# class MainWindow(QMainWindow):
#
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("My App")
#
# app = QApplication(sys.argv)
#
# window = MainWindow()
# window.show()
#
# app.exec()

#
# import sys
# from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("My App")
#
#         button = QPushButton("Press Me!")
#         button.setCheckable(True)
#         button.clicked.connect(self.the_button_was_clicked)
#
#         # Set the central widget of the Window.
#         self.setCentralWidget(button)
#
#     def the_button_was_clicked(self):
#         print("Clicked!")
#
# app = QApplication(sys.argv)
#
# window = MainWindow()
# window.show()
#
# app.exec()
#
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Press Me!")
        button.setCheckable(True)
        button.clicked.connect(self.the_button_was_clicked)
        button.clicked.connect(self.the_button_was_toggled)

        self.setCentralWidget(button)

    def the_button_was_clicked(self):
        print("Clicked!")

    def the_button_was_toggled(self, checked):
        print("Checked?", checked)

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()

# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

# "Your First QtWidgets Application"
# https://doc.qt.io/qtforpython/tutorials/basictutorial/widgets.html
import sys
from PySide6.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)
label = QLabel("Hello World!")
label.show()
app.exec()

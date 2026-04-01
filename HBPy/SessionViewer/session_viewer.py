import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget)
"""
Session Viewer — PyQt6 
======================================
Utilisable de deux façons :

  1. STANDALONE — lancer directement :
       python session_viewer.py

  2. WIDGET EMBARQUÉ — importer dans un autre script :
       from session_viewer import MoleculesListViewerWidget

       widget = MoleculesListeViewerWidget(parent=self)
       # Optionnel : brancher les messages de statut sur votre propre barre
       widget.status_message.connect(self.statusBar().showMessage)
       layout.addWidget(widget)

       # API publique disponible :
       widget.load_molecule(["a.xyz", "b.xyz"])   # charger par code
       widget.clear_all_molecules()                 # vider les tracés


Dépendances : pip install PyQt6 
"""


class MoleculesListViewerWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, apply_stylesheet: bool = True):
        """
        Parameters
        ----------
        parent : QWidget | None
            Widget parent (None = fenêtre de premier niveau).
        apply_stylesheet : bool
            Appliquer le thème sombre intégré. Mettez False si le widget
            est intégré dans une application ayant déjà son propre thème.
        """
        super().__init__(parent)
        self._color_index = 0
        self._grid_on = True
        self._legend_on = True

        self._build_ui()

        if apply_stylesheet:
            self.setStyleSheet(STYLESHEET)
            self._apply_mpl_theme()

        self._post_status("Prêt — Chargez des fichiers .dat pour commencer.")

# ═════════════════════════════════════════════════════════════════════════════
#  Enveloppe QMainWindow — mode standalone uniquement
# ═════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    Fenêtre principale pour le mode standalone.
    Enveloppe MoleculesListViewerWidget et connecte son signal status_message
    à la QStatusBar native de QMainWindow.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Molecules list Viewer")
        self.resize(1280, 780)

        # Créer le widget central
        self.viewer = MoleculesListViewerWidget(parent=self, apply_stylesheet=True)

        # Brancher le signal sur la vraie QStatusBar de la fenêtre…
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.viewer.status_message.connect(status_bar.showMessage)
        # …et masquer la barre de statut interne du widget (doublon)
        self.viewer.hide_internal_statusbar()

        self.setCentralWidget(self.viewer)

        # Appliquer le fond sombre à la fenêtre elle-même
        self.setStyleSheet("QMainWindow { background-color: #1e2127; }")


# ═════════════════════════════════════════════════════════════════════════════
#  Point d'entrée — mode standalone
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())



    

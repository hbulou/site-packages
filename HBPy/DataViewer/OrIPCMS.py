import sys
# En PyQt6, les widgets sont strictement isolés dans le module QtWidgets 
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt


import pandas as pd # Bibliothèque standard pour l'analyse de données à l'IPCMS
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtWidgets import QDialog, QComboBox, QDialogButtonBox, QFormLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# Optionnel : pour la barre d'outils (zoom, sauvegarde)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


import logging
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(funcName)s() - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


# Classe de modèle pour lier Pandas à la vue PyQt (évite l'effet boîte noire)
class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None): return self._data.shape[0]
    def columnCount(self, parent=None): return self._data.shape[1]
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None
    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._data.columns[col]
        return None



class SelectionColonnesDialog(QDialog):
    def __init__(self, colonnes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sélection des données")
        layout = QFormLayout(self)
        
        self.combo_x = QComboBox()
        self.combo_y = QComboBox()
        self.combo_x.addItems(colonnes)
        self.combo_y.addItems(colonnes)
        
        layout.addRow("Axe X :", self.combo_x)
        layout.addRow("Axe Y :", self.combo_y)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_selection(self):
        return self.combo_x.currentText(), self.combo_y.currentText()
            


class MaStationTravail(QtWidgets.QMainWindow):
    def __init__(self):
        super(MaStationTravail, self).__init__()
        
        # Chargement de l'interface XML (format ouvert et pérenne) [cite: 27]
        uic.loadUi('OrIPCMS.ui', self)

        self.dernier_df_clique = None
        
        # Configuration de l'espace de travail (MDI)
        # Note : en PyQt6, certaines constantes peuvent nécessiter l'accès via Qt.Enum
        self.mdiArea.setViewMode(QtWidgets.QMdiArea.ViewMode.SubWindowView)
        
        # Connexion des actions définies dans QtDesigner
        # Cela évite l'enfermement dans des macros propriétaires complexes [cite: 53]
        self.actionNouveauTableau.triggered.connect(self.ajouter_tableau)
        self.actionNouvellesNotes.triggered.connect(self.ajouter_notes)
        self.actionQuitter.triggered.connect(self.close)
        
        # Organisation des sous-fenêtres pour la productivité scientifique [cite: 17]
        self.actionMosaique.triggered.connect(self.mdiArea.tileSubWindows)
        self.actionCascade.triggered.connect(self.mdiArea.cascadeSubWindows)

        # --- CONNEXIONS ---
        # On connecte l'action créée dans QtDesigner à notre fonction Python
        self.actionTracer_un_graphique.triggered.connect(self.preparer_trace)

    def preparer_trace(self):
        """Récupère le tableau de la fenêtre active et lance le dialogue de sélection."""
        active_sub = self.mdiArea.activeSubWindow()
        
        if active_sub and isinstance(active_sub.widget(), QtWidgets.QTableView):
            # On récupère le DataFrame stocké dans notre PandasModel
            # Note : On accède à ._data que nous avons défini dans la classe PandasModel
            df = active_sub.widget().model()._data
            self.tracer_graphique(df)
        else:
            QtWidgets.QMessageBox.warning(self, "Action requise", 
                                        "Veuillez cliquer sur une fenêtre de tableau avant de tracer.")



    # def tracer_graphique(self, df):
    #     dialog = SelectionColonnesDialog(df.columns.tolist(), self)

    #     if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
    #         col_x, col_y = dialog.get_selection()

    #         # 1. Création de la figure (le fond) et des axes (le graphique)
    #         fig = Figure(figsize=(6, 4), layout='constrained')
    #         canvas = FigureCanvas(fig)
    #         # Autoriser le menu contextuel personnalisé
    #         canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    #         canvas.customContextMenuRequested.connect(lambda pos: self.ouvrir_menu_graphique(pos, canvas, df))

    #         # 2. Créer le widget conteneur (C'EST ICI QU'ON DÉFINIT LA VARIABLE)
    #         conteneur = QtWidgets.QWidget()
    #         # Pour pouvoir retrouver le canvas facilement plus tard (comme suggéré précédemment)
    #         conteneur.canvas = canvas
    #         ax = fig.add_subplot(111)

    #         # 2. Le tracé scientifique (Vérifiable et précis)
    #         ax.plot(df[col_x], df[col_y], marker='o', linestyle='-', label=f"{col_y}")
    #         ax.set_xlabel(col_x)
    #         ax.set_ylabel(col_y)
    #         ax.legend()
    #         ax.grid(True)

    #         # 3. Création d'un conteneur pour inclure le graphique ET la barre d'outils
    #         conteneur = QtWidgets.QWidget()
    #         layout_v = QtWidgets.QVBoxLayout(conteneur)
    #         layout_v.addWidget(canvas)

    #         # Ajout de la barre d'outils standard (Zoom, Pan, Save)
    #         toolbar = NavigationToolbar(canvas, conteneur)
    #         layout_v.addWidget(toolbar)

    #         # 4. Affichage dans une nouvelle sous-fenêtre MDI
    #         sub = QtWidgets.QMdiSubWindow()
    #         sub.setWidget(conteneur)
    #         sub.setWindowTitle(f"Visualisation : {col_y}")
    #         self.mdiArea.addSubWindow(sub)
    #         sub.show()

    def tracer_graphique(self, df):
        dialog = SelectionColonnesDialog(df.columns.tolist(), self)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            col_x, col_y = dialog.get_selection()

            fig = Figure(figsize=(6, 4), layout='constrained')
            canvas = FigureCanvas(fig)
            
            # Connexion du menu contextuel
            canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            canvas.customContextMenuRequested.connect(lambda pos: self.ouvrir_menu_graphique(pos, canvas, df))

            # Création propre du conteneur
            conteneur = QtWidgets.QWidget()
            conteneur.canvas = canvas # Stockage de la référence pour le clic droit
            
            layout_v = QtWidgets.QVBoxLayout(conteneur)
            layout_v.addWidget(canvas)
            
            # Ajout de la barre d'outils (Navigation)
            toolbar = NavigationToolbar(canvas, conteneur)
            layout_v.addWidget(toolbar)

            # Dessin initial
            ax = fig.add_subplot(111)
            ax.plot(df[col_x], df[col_y], marker='o', label=f"{col_y}")
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.legend()
            ax.grid(True)

            sub = QtWidgets.QMdiSubWindow()
            sub.setWidget(conteneur)
            self.mdiArea.addSubWindow(sub)
            sub.show()
    
    def ajouter_courbe_a_existant(self, conteneur, df):
        """Ajoute une courbe sur un graphique déjà ouvert."""
        # Sélection des colonnes dans le nouveau DataFrame
        dialog = SelectionColonnesDialog(df.columns.tolist(), self)
        
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            col_x, col_y = dialog.get_selection()
            
            # Récupération du canvas et de l'axe via la référence stockée dans le conteneur
            canvas = conteneur.canvas
            ax = canvas.figure.axes[0]
            
            # Ajout de la nouvelle courbe (tracé vérifiable et précis)
            ax.plot(df[col_x], df[col_y], marker='s', linestyle='--', label=f"Add: {col_y}")
            ax.legend()
            
            # Rafraîchissement indispensable du dessin
            canvas.draw()
    def ouvrir_menu_graphique(self, position, canvas, df_initial):
        """Génère le menu contextuel sur le graphique."""
        menu = QtWidgets.QMenu(self)
        
        # Action pour superposer des données (comme dans Origin Pro)
        action_ajouter = menu.addAction("Ajouter une courbe depuis le tableau actif")
        
        # Action choisie à la position de la souris
        action_executee = menu.exec(canvas.mapToGlobal(position))
        
        if action_executee == action_ajouter:
            # On utilise votre méthode de récupération pour trouver les données
            df_a_utiliser = self.recuperer_df_pour_ajout()
            
            if df_a_utiliser is not None:
                # Le parentWidget du canvas est notre 'conteneur' créé dans tracer_graphique
                self.ajouter_courbe_a_existant(canvas.parentWidget(), df_a_utiliser)
            else:
                QtWidgets.QMessageBox.warning(self, "Erreur", "Aucun tableau de données sélectionné.")
            
    def ajouter_tableau(self):
        # 1. Sélection du fichier (Souveraineté : contrôle total sur les sources de données)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Ouvrir un fichier de données", "", "Fichiers de données (*.dat *.txt *.csv)"
        )

        if file_path:
            try:
                # 2. Lecture du fichier .dat via Pandas (Format ouvert et pérenne )
                # On suppose ici un séparateur tabulation ou espace fréquent en physique
                df = pd.read_csv(file_path, sep=r'\s+', engine='python',comment='#',header=None)

                # 3. Affichage dans l'espace MDI
                sub = QtWidgets.QMdiSubWindow()
                table = QtWidgets.QTableView()
                model = PandasModel(df)
                table.setModel(model)
                # --- ASTUCE STRATÉGIQUE ---
                # On stocke le DF dans l'objet table pour le retrouver facilement
                table.df_interne = df 
                # On connecte le clic sur le tableau pour mettre à jour la sélection
                table.clicked.connect(lambda: self.set_dernier_df(table.df_interne))
                
                sub.setWidget(table)
                sub.setWindowTitle(f"Données : {file_path.split('/')[-1]}")
                self.mdiArea.addSubWindow(sub)
                sub.show()

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Erreur de lecture", f"Impossible de lire le fichier : {e}")
    def set_dernier_df(self, df):
        self.dernier_df_clique = df
        logger.info(df)


    def recuperer_df_pour_ajout(self):
        """
        Priorité 1 : Le dernier tableau sur lequel on a cliqué.
        Priorité 2 : Si aucun clic, le tableau de la fenêtre active.
        """
        if self.dernier_df_clique is not None:
            return self.dernier_df_clique

        # Sécurité : si l'utilisateur n'a pas cliqué, on cherche la fenêtre active
        active_sub = self.mdiArea.activeSubWindow()
        if active_sub and isinstance(active_sub.widget(), QtWidgets.QTableView):
            return active_sub.widget().model()._data

        return None
    
    def ajouter_notes(self):
        """Espace de rédaction (compatible avec le format LaTeX préconisé)[cite: 47, 49]."""
        sub = QtWidgets.QMdiSubWindow()
        editeur = QtWidgets.QTextEdit()
        sub.setWidget(editeur)
        sub.setWindowTitle("Notes de Manipulation (LaTeX)")
        self.mdiArea.addSubWindow(sub)
        sub.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # L'utilisation de PyQt6 sous Linux garantit l'absence de télémétrie [cite: 21]
    window = MaStationTravail()
    window.show()
    sys.exit(app.exec()) # En PyQt6, exec_() devient exec()

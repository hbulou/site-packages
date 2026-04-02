import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget,QFileDialog
from PyQt6.QtCore import Qt

from HBPy.Molecule.molecule_session import MoleculeSession
from HBPy.SessionViewer.HBPy_Session_Widgets import Ui_MoleculeManager
from HBPy.SessionViewer.MainWindow import Ui_MainWindow
from HBPy.QtWidgets.molecule_viewer import Ui_MoleculeViewer
import pandas as pd
from HBPy.PandasModel.pandasmodel import PandasModel

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtCore import pyqtSignal

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdmolfiles import MolToXYZFile

from HBPy.Molecule.Crystal import Crystal
from HBPy.Molecule.molecule_session import MoleculeEntry
import tempfile
import os


import logging
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Fenêtres secondaires ────────────────────────────────────────────────────────
class MoleculeManagerWindow(QMainWindow, Ui_MoleculeManager):
    signal_ferme = pyqtSignal()   # signal émis à la fermeture
    signal_delete_molecule = pyqtSignal(int) # ← nouveau signal, émet le numéro de ligne
    signal_focus_molecule = pyqtSignal(int) # ← nouveau signal, émet le numéro de ligne
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)                          # applique le .ui
        self.setWindowFlags(Qt.WindowType.Window)   # fenêtre indépendante
        self.molecule_list_pandas_model = PandasModel(pd.DataFrame())
        self.WD_molecules_list.setModel(self.molecule_list_pandas_model)  # ← une seule fois ici
        self.WD_molecules_list.setAlternatingRowColors(True)
        self.WD_molecules_list.horizontalHeader().setStretchLastSection(True)
        # ── Menu contextuel ───────────────────────────────────────────
        self.WD_molecules_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.WD_molecules_list.customContextMenuRequested.connect(self._open_menu)

    def closeEvent(self, event: QCloseEvent):
        self.signal_ferme.emit()  # prévenir la fenêtre principale
        super().closeEvent(event)
    def _delete(self, row):
        from PyQt6.QtWidgets import QMessageBox
        reponse = QMessageBox.question(
            self, "Detete",
            f"Supprimer la molécule ligne {row} ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reponse == QMessageBox.StandardButton.Yes:
            print(f"Supprimer ligne {row} id {self.molecule_list_pandas_model}")
            self.signal_delete_molecule.emit(row)   # ← délègue à MainApp
        # df = self.session.to_dataframe()
    
        # ligne = df.iloc[row]          # récupère la ligne cliquée
        # mol_id = ligne["id"]          # récupère l'id
        # source = ligne["source"]      # récupère le nom du fichier
        # logger.info(f"Suppression de {source} (id={mol_id})")
        # self.session.delete(mol_id)
        # self.MoleculeManager.update(self.session.to_dataframe())
        # appeler self.session.supprimer(row) si vous l'implémentez

    def _open_menu(self, position):
        # Récupérer la ligne cliquée
        index = self.WD_molecules_list.indexAt(position)
        if not index.isValid():
            return  # clic dans le vide → rien

        # Récupérer le numéro de ligne
        row = index.row()

        # Construire le menu
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)

        #action_charger    = menu.addAction("📂  Charger dans le viewer")
        #action_renommer   = menu.addAction("✏️  Renommer")
        action_focus   = menu.addAction("✏️  focus")
        menu.addSeparator()
        action_delete  = menu.addAction("🗑  Delete")

        # Afficher le menu à la position de la souris
        action_choisie = menu.exec(self.WD_molecules_list.viewport().mapToGlobal(position))

        if action_choisie == action_delete:
            self._delete(row)
        elif action_choisie == action_focus:
            self.signal_focus_molecule.emit(row)   # ← délègue à MainApp
        # Déclencher l'action choisie
        #if action_choisie == action_charger:
        #    self._charger(row)
        #elif action_choisie == action_renommer:
        #    self._renommer(row)
        #elif action_choisie == action_supprimer:
        #    self._supprimer(row)
    def update(self, df):
        """Appelée depuis MainApp à chaque chargement de molécule."""
        self.molecule_list_pandas_model.setDataFrame(df)
        self.WD_molecules_list.resizeColumnsToContents()

        
class MoleculeViewerWindow(QMainWindow, Ui_MoleculeViewer):
    signal_ferme = pyqtSignal()   # signal émis à la fermeture
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)                          # applique le .ui
        self.setWindowFlags(Qt.WindowType.Window)   # fenêtre indépendante

    def closeEvent(self, event: QCloseEvent):
        self.signal_ferme.emit()  # prévenir la fenêtre principale
        super().closeEvent(event)


# ── Fenêtre principale ────────────────────────────────────────────────────────
class MainApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.session=MoleculeSession()
        # _______________________________________________________
        # Créer la fenêtre MoleculeManager (pas encore visible)
        # _______________________________________________________
        self.MoleculeManager = MoleculeManagerWindow(parent=None)
        self.MoleculeViewer  = MoleculeViewerWindow(parent=None)
        self.windows_list=[self.MoleculeManager,self.MoleculeViewer]

        # bouton pour afficher/cacher MoleculeManagerWindow
        self.WD_Btn_Molecule_Manager.clicked.connect(self._toggle_molecule_manager)
        self.WD_Btn_Molecule_Viewer.clicked.connect(self._toggle_molecule_viewer)
        # pour connecter le choix de la molecule à détruite avec
        # session
        self.MoleculeManager.signal_delete_molecule.connect(self._on_delete_molecule)
        self.MoleculeManager.signal_focus_molecule.connect(self._on_focus_molecule)

        self.MoleculeManager.signal_ferme.connect(
            lambda: print("Fenêtre fermée via la croix")
            # ou toute autre action à synchroniser
        )

        self.MoleculeManager.WD_Btn_load_molecule.clicked.connect(self.load_model)
        self.MoleculeManager.WD_Btn_alkyl_builder.clicked.connect(self.build_alkyl_chain)

    def _on_focus_molecule(self,row):
        # ── Affichage dans le MoleculeViewer ──────────────────────────────
        self.MoleculeViewer.MoleculeDisplayArea.set_molecule(self.session.molecules[row].molecule)
        
    def _on_delete_molecule(self, row):
        df = self.session.to_dataframe()
        mol_id = df.iloc[row]["id"]
        source = df.iloc[row]["source"]
        logger.info(f"Suppression de {source} (id={mol_id})")
        self.session.delete(mol_id)
        self.MoleculeManager.update(self.session.to_dataframe())        
    def _toggle_molecule_manager(self):
        if self.MoleculeManager.isVisible():
            self.MoleculeManager.hide()
        else:
            self.MoleculeManager.show()
            self.MoleculeManager.raise_()        # passer au premier plan
            self.MoleculeManager.activateWindow() # donner le focus
            # Récupérer la position et la taille de la fenêtre principale
            # geo = self.geometry()
            # frameGeometry() inclut la bordure et la barre de titre
            geo = self.frameGeometry()
            # Placer la fenêtre secondaire juste à droite
            x = geo.x() -self.MoleculeManager.width()
            y = geo.y()
            self.MoleculeManager.move(x, y)

    def _toggle_molecule_viewer(self):
        if self.MoleculeViewer.isVisible():
            self.MoleculeViewer.hide()
        else:

            self.MoleculeViewer.show()
            self.MoleculeViewer.raise_()        # passer au premier plan
            self.MoleculeViewer.activateWindow() # donner le focus
            geo = self.frameGeometry()
            # Placer la fenêtre secondaire juste à droite
            x = geo.x() 
            y = geo.y()+ geo.height()
            self.MoleculeViewer.move(x, y)

    def closeEvent(self, event: QCloseEvent):
        # Fermer toutes les fenêtres secondaires d'un coup
        for fenetre in self.windows_list:
            fenetre.close()
        super().closeEvent(event)
    def load_model(self,filename=None):
        # on choisit le fichier *.xyz
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Choisir un fichier",
                "",  # répertoire de départ ("" = courant)
                "Fichiers xyz (*.xyz);;Tous les fichiers (*)"
            )
        if not filename:
            return  # Sortir si vide
        else:
            self.session.load(filename=filename)
            logger.info(f"{self.session.to_dataframe()}")
            # ── Affichage dans le MoleculeViewer ──────────────────────────────
            self.MoleculeViewer.MoleculeDisplayArea.set_molecule(self.session.molecules[-1].molecule)
            # ── Affichage dans le MoleculeManager ────────────────────────────
            self.MoleculeManager.update(self.session.to_dataframe())

            #self.WD_molecules_list.setEditTriggers(                              
            #    QAbstractItemView.EditTrigger.DoubleClicked |  # Double-clic pour éditer
            #    QAbstractItemView.EditTrigger.SelectedClicked   # Ou clic sur sélection
            #)
            #self.atom_model.dataModified.connect(self.on_atom_data_modified)
    def mol_to_crystal(self,mol) -> Crystal:
        """Convertit un rdkit.Chem.Mol en Crystal via un fichier xyz temporaire."""
    
        # 1. Écrire le Mol dans un fichier xyz temporaire
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            tmp_path = tmp.name

        MolToXYZFile(mol, tmp_path)

        # 2. Charger le fichier xyz dans un Crystal
        crystal = Crystal()
        crystal.load_file(tmp_path)
        crystal.MassCenter()
        crystal.get_element_distribution()
        crystal.get_structure()
    
        # 3. Supprimer le fichier temporaire
        os.remove(tmp_path)

        return crystal
    def build_alkyl_chain(self):
        smiles = self.MoleculeManager.WD_LE_smile.text()
                # 1. On crée la molécule parfaite avec tous ses H
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    
        # 2. On génère la 3D de base (qui risque d'être tordue)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
    
        # 3. LE SECRET : Forcer les carbones aliphatiques à s'aligner
        conformer = mol.GetConformer()
        
        # On isole uniquement les carbones qui ne font pas partie d'un cycle (ta chaîne)
        carbones_chaine = [atom.GetIdx() for atom in mol.GetAtoms() 
                           if atom.GetSymbol() == 'C' and not atom.IsInRing()]
    
        # On impose un angle dièdre de 180° (trans) pour chaque groupe de 4 carbones
        for i in range(len(carbones_chaine) - 3):
            c1 = carbones_chaine[i]
            c2 = carbones_chaine[i+1]
            c3 = carbones_chaine[i+2]
            c4 = carbones_chaine[i+3]
            try:
                rdMolTransforms.SetDihedralDeg(conformer, c1, c2, c3, c4, 180.0)
            except ValueError:
                pass # Ignore poliment si les atomes ne sont pas directement liés

        # 4. On détend la molécule (Force Field MMFF) pour optimiser les liaisons C-H
        # sans casser notre beau zig-zag tout neuf !
        AllChem.MMFFOptimizeMolecule(mol)
        print(type(mol),dir(mol))
        # Conversion directe en Crystal
        crystal = self.mol_to_crystal(mol)

        # Ajout à la session
        self.session.molecules.append(
            MoleculeEntry(len(self.session.molecules), "from_smiles", crystal)
        )
        self.MoleculeViewer.MoleculeDisplayArea.set_molecule(self.session.molecules[-1].molecule)
        self.MoleculeManager.update(self.session.to_dataframe())
        #self.session.add_molecule(source=smiles,molecule=mol)
        #self.MoleculeViewer.MoleculeDisplayArea.set_molecule(self.session.molecules[-1].molecule)
        #self.MoleculeManager.update(self.session.to_dataframe())
        
# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())

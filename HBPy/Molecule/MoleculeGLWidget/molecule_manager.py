"""
molecule_manager.py
-------------------
Fenêtre indépendante de gestion des molécules.
Communique avec MainApp via des signaux PyQt6.

Intégration dans Py_Imidazolium.py :
    1. from molecule_manager import MoleculeManagerWindow
    2. Dans __init__ de MainApp :
           self.mol_manager = MoleculeManagerWindow()
           self.mol_manager.signal_charger_molecule.connect(self.load_model)
           self.mol_manager.signal_supprimer_molecule.connect(self.on_molecule_supprimee)
           self.WD_Btn_gestionnaire.clicked.connect(self.mol_manager.show)   # bouton à ajouter dans le .ui
    3. Après chaque load_model réussi, appeler :
           self.mol_manager.ajouter_molecule(self.filename)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QLabel,
    QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from pathlib import Path
import uuid


# ─────────────────────────────────────────────────────────────────────────────
class MoleculeManagerWindow(QWidget):
    """
    Fenêtre flottante indépendante de la fenêtre principale.

    Signaux émis vers MainApp :
        signal_charger_molecule(str)   → chemin du fichier .xyz à charger
        signal_supprimer_molecule(str) → identifiant unique de la molécule supprimée
    """

    signal_charger_molecule  = pyqtSignal(str)   # filename
    signal_supprimer_molecule = pyqtSignal(str)   # mol_id

    # Colonnes du tableau
    COL_NOM     = 0
    COL_FICHIER = 1
    COL_STATUT  = 2
    COL_ID      = 3   # colonne cachée → stocke l'UUID interne

    # ── Initialisation ────────────────────────────────────────────────────────
    def __init__(self, parent=None):
        super().__init__(parent)

        # Fenêtre indépendante (pas de parent visuel)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Gestionnaire de molécules")
        self.setMinimumSize(620, 380)
        self.resize(700, 420)

        # Dictionnaire interne  {mol_id: {"nom": str, "fichier": str, "statut": str}}
        self._molecules: dict[str, dict] = {}

        self._build_ui()
        self._apply_style()

    # ── Construction de l'interface ───────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # --- Titre ---
        titre = QLabel("🧪 Gestionnaire de molécules")
        titre.setObjectName("titre")
        layout.addWidget(titre)

        # --- Barre de boutons ---
        barre = QHBoxLayout()
        barre.setSpacing(8)

        self.btn_nouvelle  = QPushButton("➕  Nouvelle molécule")
        self.btn_supprimer = QPushButton("🗑  Supprimer")
        self.btn_charger   = QPushButton("📂  Charger dans la vue")

        self.btn_nouvelle.setObjectName("btn_action")
        self.btn_supprimer.setObjectName("btn_danger")
        self.btn_charger.setObjectName("btn_action")

        barre.addWidget(self.btn_nouvelle)
        barre.addWidget(self.btn_charger)
        barre.addStretch()
        barre.addWidget(self.btn_supprimer)
        layout.addLayout(barre)

        # --- Tableau ---
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Nom", "Fichier .xyz", "Statut", "_id"])
        self.table.setColumnHidden(self.COL_ID, True)          # UUID caché

        # Comportement du tableau
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)

        # Redimensionnement des colonnes
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.COL_NOM,     QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(self.COL_FICHIER, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_STATUT,  QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(self.COL_NOM, 160)

        layout.addWidget(self.table)

        # --- Aide contextuelle ---
        aide = QLabel("💡 Double-cliquez sur le nom pour le modifier.")
        aide.setObjectName("aide")
        layout.addWidget(aide)

        # --- Connexions ---
        self.btn_nouvelle.clicked.connect(self._nouvelle_molecule)
        self.btn_supprimer.clicked.connect(self._supprimer_molecule)
        self.btn_charger.clicked.connect(self._charger_dans_vue)
        self.table.itemChanged.connect(self._on_nom_modifie)

    # ── Style ─────────────────────────────────────────────────────────────────
    def _apply_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', 'Ubuntu', sans-serif;
                font-size: 13px;
            }
            QLabel#titre {
                font-size: 16px;
                font-weight: bold;
                color: #89b4fa;
                padding-bottom: 4px;
            }
            QLabel#aide {
                color: #6c7086;
                font-size: 11px;
            }
            QPushButton {
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
                border: none;
            }
            QPushButton#btn_action {
                background-color: #313244;
                color: #cdd6f4;
            }
            QPushButton#btn_action:hover {
                background-color: #45475a;
            }
            QPushButton#btn_action:pressed {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QPushButton#btn_danger {
                background-color: #313244;
                color: #f38ba8;
            }
            QPushButton#btn_danger:hover {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
            QTableWidget {
                background-color: #181825;
                alternate-background-color: #1e1e2e;
                gridline-color: #313244;
                border: 1px solid #313244;
                border-radius: 6px;
            }
            QTableWidget::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QHeaderView::section {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
                border: none;
                padding: 5px 8px;
            }
        """)

    # ── API publique (appelée depuis MainApp) ─────────────────────────────────
    def ajouter_molecule(self, filename: str, nom: str = "") -> str:
        """
        Ajoute une molécule au gestionnaire depuis MainApp.

        Paramètres :
            filename : chemin vers le fichier .xyz
            nom      : nom affiché (si vide → déduit du nom de fichier)

        Retourne :
            mol_id   : identifiant unique de la molécule (UUID)
        """
        mol_id = str(uuid.uuid4())
        if not nom:
            nom = Path(filename).stem   # ex: "xtbopt" depuis "xtbopt.xyz"

        self._molecules[mol_id] = {
            "nom":     nom,
            "fichier": filename,
            "statut":  "✅ Chargée",
        }
        self._ajouter_ligne(mol_id)
        self.show()      # affiche la fenêtre si elle était cachée
        self.raise_()    # la met au premier plan
        return mol_id

    def definir_statut(self, mol_id: str, statut: str):
        """Met à jour le statut d'une molécule dans le tableau."""
        if mol_id not in self._molecules:
            return
        self._molecules[mol_id]["statut"] = statut
        self._rafraichir_ligne(mol_id)

    # ── Slots privés ──────────────────────────────────────────────────────────
    def _nouvelle_molecule(self):
        """Ouvre un QFileDialog pour choisir un .xyz et l'ajoute à la liste."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir un fichier molécule",
            "",
            "Fichiers XYZ (*.xyz);;Tous (*)"
        )
        if filename:
            mol_id = self.ajouter_molecule(filename)
            # Émettre le signal pour que MainApp charge la molécule
            self.signal_charger_molecule.emit(filename)

    def _supprimer_molecule(self):
        """Supprime la ligne sélectionnée après confirmation."""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Info", "Sélectionnez d'abord une molécule.")
            return

        nom = self.table.item(row, self.COL_NOM).text()
        reponse = QMessageBox.question(
            self,
            "Confirmer la suppression",
            f"Supprimer « {nom} » de la liste ?\n(Le fichier .xyz n'est pas effacé.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reponse != QMessageBox.StandardButton.Yes:
            return

        mol_id = self.table.item(row, self.COL_ID).text()
        self._molecules.pop(mol_id, None)
        self.table.removeRow(row)
        self.signal_supprimer_molecule.emit(mol_id)

    def _charger_dans_vue(self):
        """Émet le signal pour charger la molécule sélectionnée dans MainApp."""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Info", "Sélectionnez d'abord une molécule.")
            return

        mol_id   = self.table.item(row, self.COL_ID).text()
        filename = self._molecules[mol_id]["fichier"]
        self.signal_charger_molecule.emit(filename)
        self.definir_statut(mol_id, "🔵 Active")

    def _on_nom_modifie(self, item: QTableWidgetItem):
        """Met à jour le dictionnaire interne quand l'utilisateur renomme."""
        if item.column() != self.COL_NOM:
            return
        row    = item.row()
        id_item = self.table.item(row, self.COL_ID)
        if id_item is None:
            return
        mol_id = id_item.text()
        if mol_id in self._molecules:
            self._molecules[mol_id]["nom"] = item.text()

    # ── Helpers d'affichage ───────────────────────────────────────────────────
    def _ajouter_ligne(self, mol_id: str):
        """Insère une nouvelle ligne dans le tableau."""
        data   = self._molecules[mol_id]
        row    = self.table.rowCount()
        self.table.blockSignals(True)   # évite de déclencher _on_nom_modifie
        self.table.insertRow(row)

        # Colonne Nom (éditable)
        item_nom = QTableWidgetItem(data["nom"])
        item_nom.setFlags(item_nom.flags() | Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, self.COL_NOM, item_nom)

        # Colonne Fichier (non éditable)
        item_fic = QTableWidgetItem(str(Path(data["fichier"]).name))
        item_fic.setFlags(item_fic.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item_fic.setToolTip(data["fichier"])   # chemin complet en tooltip
        self.table.setItem(row, self.COL_FICHIER, item_fic)

        # Colonne Statut (non éditable)
        item_sta = QTableWidgetItem(data["statut"])
        item_sta.setFlags(item_sta.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item_sta.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, self.COL_STATUT, item_sta)

        # Colonne ID cachée
        item_id = QTableWidgetItem(mol_id)
        self.table.setItem(row, self.COL_ID, item_id)

        self.table.blockSignals(False)
        self.table.selectRow(row)

    def _rafraichir_ligne(self, mol_id: str):
        """Met à jour uniquement la cellule Statut d'une molécule existante."""
        for row in range(self.table.rowCount()):
            if self.table.item(row, self.COL_ID).text() == mol_id:
                self.table.blockSignals(True)
                self.table.item(row, self.COL_STATUT).setText(
                    self._molecules[mol_id]["statut"]
                )
                self.table.blockSignals(False)
                break

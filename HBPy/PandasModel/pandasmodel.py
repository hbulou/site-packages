# pip install PyQt6 pandas
#import sys
import pandas as pd
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant,pyqtSignal
#from PyQt6.QtWidgets import QApplication, QMainWindow, QTableView, QWidget, QVBoxLayout
#from PyQt6.QtGui import QAction

class PandasModel(QAbstractTableModel):
    """
    Modèle Qt qui expose un pandas DataFrame à un QTableView.
    Supporte l'affichage, les en-têtes et le tri.
    """
    dataModified = pyqtSignal(dict) 
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        # on copie pour éviter les surprises si df est modifié ailleurs
        self._df = df.copy()


    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        """
        CETTE MÉTHODE EST APPELÉE AUTOMATIQUEMENT PAR Qt
            
        Args:
            index: QModelIndex - position de la cellule (ligne, colonne)
            value: str - nouvelle valeur tapée par l'utilisateur (toujours string)
            role: EditRole - indique que c'est une édition
            
        Returns:
            bool - True si modification réussie, False sinon
            """
        print(f"PandasModel() -> SetData()")
        row=index.row()
        col=index.column()
        print(f"modification de la cellule ({row,col} : {value}")
        print(self._df.iloc[row])
        print(self._df.iloc[row,0])
        # 1️⃣ METTRE À JOUR LE DATAFRAME
        self._df.iat[index.row(), index.column()] = value  # ← LIGNE MANQUANTE !
        
        # 6. ÉMETTRE LE SIGNAL pour notifier votre code
        self.dataModified.emit({'idx':self._df.iloc[row,0],'row':index.row(),'col':index.column(),'val':value})
        return True


    # --- dimensions ---
    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._df.shape[1]

    # --- données cellules ---
    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            val = self._df.iat[index.row(), index.column()]
            # Affichage propre pour NaN
            if pd.isna(val):
                return ""
            return str(val)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            # nombres à droite, texte à gauche
            col = self._df.columns[index.column()]
            if pd.api.types.is_numeric_dtype(self._df[col]):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        return QVariant()


    def flags(self, index):
        """
        Rendre les cellules éditables
        """
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
    # --- entêtes ---
    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        else:
            # Index de ligne (1-based plus lisible)
            return str(section + 1)

    # --- tri par clic d'en-tête ---
    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder):
        # Rien à trier si DF vide ou si index de colonne invalide
        if getattr(self, "_df", None) is None or self._df.empty:
            return
        if column < 0 or column >= self._df.shape[1]:
            return

        self.layoutAboutToBeChanged.emit()
        colname = self._df.columns[column]
        ascending = (order == Qt.SortOrder.AscendingOrder)
        try:
            # tri stable + NaN en bas, puis réindex propre
            self._df = (
                self._df.sort_values(by=colname,
                                     ascending=ascending,
                                     kind="mergesort",
                                     na_position="last")
                .reset_index(drop=True)
            )
        finally:
            self.layoutChanged.emit()


    # --- utilitaire pour changer dynamiquement de DataFrame ---
    def setDataFrame(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()



"""
DAT File Viewer — PyQt6 + Matplotlib
======================================
Utilisable de deux façons :

  1. STANDALONE — lancer directement :
       python dat_viewer.py

  2. WIDGET EMBARQUÉ — importer dans un autre script :
       from dat_viewer import DatViewerWidget

       widget = DatViewerWidget(parent=self)
       # Optionnel : brancher les messages de statut sur votre propre barre
       widget.status_message.connect(self.statusBar().showMessage)
       layout.addWidget(widget)

       # API publique disponible :
       widget.load_files(["a.dat", "b.dat"])   # charger par code
       widget.clear_all_plots()                 # vider les tracés
       files = widget.dat_files                 # liste des DatFile chargés

Dépendances : pip install PyQt6 matplotlib numpy
"""

import sys
import os
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QListWidget, QListWidgetItem, QLabel,
    QFileDialog, QTextEdit, QGroupBox, QDialog, QComboBox,
    QCheckBox, QToolBar, QMessageBox, QAbstractItemView, QColorDialog,
    QDoubleSpinBox, QFormLayout, QLineEdit, QStatusBar, QSizePolicy,
)
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtCore import Qt, QSize, pyqtSignal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ─────────────────────────────────────────────────────────────────────────────
# Palette de couleurs cyclique
# ─────────────────────────────────────────────────────────────────────────────
PLOT_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#E91E63", "#3F51B5",
]

STYLESHEET = """
    QWidget {
        background-color: #1e2127;
        color: #abb2bf;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        font-size: 10pt;
    }
    QGroupBox {
        border: 1px solid #3e4451;
        border-radius: 6px;
        margin-top: 10px;
        padding: 6px;
        font-weight: bold;
        color: #61afef;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
    }
    QListWidget {
        background-color: #21252b;
        border: 1px solid #3e4451;
        border-radius: 4px;
        color: #abb2bf;
    }
    QListWidget::item:selected {
        background-color: #2c3e50;
        color: #61afef;
    }
    QListWidget::item:hover { background-color: #2a2f3a; }
    QPushButton {
        background-color: #2c313a;
        border: 1px solid #3e4451;
        border-radius: 4px;
        padding: 4px 12px;
        color: #abb2bf;
    }
    QPushButton:hover {
        background-color: #3a3f4b;
        border-color: #61afef;
        color: #61afef;
    }
    QPushButton:pressed { background-color: #1a1d24; }
    QPushButton:checked {
        background-color: #264f78;
        border-color: #61afef;
        color: #61afef;
    }
    QTextEdit {
        background-color: #21252b;
        border: 1px solid #3e4451;
        border-radius: 4px;
        color: #98c379;
    }
    QToolBar {
        background-color: #21252b;
        border-bottom: 1px solid #3e4451;
        spacing: 4px;
        padding: 2px;
    }
    QToolBar QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 3px 8px;
        color: #abb2bf;
    }
    QToolBar QToolButton:hover {
        background-color: #2c313a;
        border-color: #3e4451;
    }
    QToolBar QToolButton:checked {
        background-color: #264f78;
        border-color: #61afef;
        color: #61afef;
    }
    QStatusBar {
        background-color: #21252b;
        color: #5c6370;
        border-top: 1px solid #3e4451;
    }
    QSplitter::handle { background-color: #3e4451; }
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
        background-color: #21252b;
        border: 1px solid #3e4451;
        border-radius: 4px;
        padding: 2px 6px;
        color: #abb2bf;
    }
    QDialog { background-color: #1e2127; }
    QLabel#status_label {
        color: #5c6370;
        font-size: 9pt;
        padding: 2px 6px;
        border-top: 1px solid #3e4451;
        background-color: #21252b;
    }
"""


# ═════════════════════════════════════════════════════════════════════════════
# Modèles de données  (aucune dépendance Qt — réutilisables partout)
# ═════════════════════════════════════════════════════════════════════════════

class DatFile:
    """Représente un fichier .dat chargé en mémoire."""

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.comments: list[str] = []
        self.headers: list[str] = []
        self.data: np.ndarray | None = None
        self.load_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._parse()

    def _parse(self):
        rows = []
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    self.comments.append(stripped[1:].strip())
                else:
                    parts = stripped.replace(",", " ").replace(";", " ").replace("\t", " ").split()
                    try:
                        rows.append([float(p) for p in parts])
                    except ValueError:
                        self.headers = parts
        if rows:
            self.data = np.array(rows)

    @property
    def n_rows(self):
        return self.data.shape[0] if self.data is not None else 0

    @property
    def n_cols(self):
        return self.data.shape[1] if self.data is not None else 0

    @property
    def col_names(self) -> list[str]:
        if self.headers and len(self.headers) == self.n_cols:
            return self.headers
        return ["X"] + [f"Y{i}" for i in range(1, self.n_cols)]

    def info_text(self) -> str:
        lines = [
            f"📄 Fichier   : {self.name}",
            f"📁 Chemin    : {self.path}",
            f"🕐 Chargé le : {self.load_time}",
            f"📏 Taille    : {os.path.getsize(self.path):,} octets",
            "",
            f"📊 Lignes de données : {self.n_rows}",
            f"📊 Colonnes          : {self.n_cols}",
        ]
        if self.col_names:
            lines.append(f"📊 Noms colonnes     : {', '.join(self.col_names)}")
        if self.data is not None and self.n_rows > 0:
            lines += ["", "📈 Statistiques par colonne :"]
            for i, cn in enumerate(self.col_names):
                col = self.data[:, i]
                lines.append(
                    f"   {cn:>8s} — min={col.min():.4g}  max={col.max():.4g}"
                    f"  moy={col.mean():.4g}  σ={col.std():.4g}"
                )
        if self.comments:
            lines += ["", "💬 Commentaires :"] + [f"   {c}" for c in self.comments]
        return "\n".join(lines)


class PlotEntry:
    """Décrit une courbe à tracer."""
    _counter = 0

    def __init__(self, dat_file: DatFile, x_col: int, y_col: int,
                 color: str, label: str, linewidth: float = 1.5,
                 linestyle: str = "-", marker: str = "None"):
        PlotEntry._counter += 1
        self.id = PlotEntry._counter
        self.dat_file = dat_file
        self.x_col = x_col
        self.y_col = y_col
        self.color = color
        self.label = label
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.marker = marker
        self.visible = True
        self.line_object = None


# ═════════════════════════════════════════════════════════════════════════════
# Boîte de dialogue : ajouter un tracé
# ═════════════════════════════════════════════════════════════════════════════

class AddPlotDialog(QDialog):
    def __init__(self, dat_files: list[DatFile], color_index: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajouter un tracé")
        self.setMinimumWidth(420)
        self.dat_files = dat_files
        self._chosen_color = PLOT_COLORS[color_index % len(PLOT_COLORS)]
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.file_combo = QComboBox()
        for df in self.dat_files:
            self.file_combo.addItem(df.name, df)
        self.file_combo.currentIndexChanged.connect(self._on_file_changed)
        form.addRow("Fichier :", self.file_combo)

        self.x_combo = QComboBox()
        form.addRow("Colonne X :", self.x_combo)

        self.y_combo = QComboBox()
        form.addRow("Colonne Y :", self.y_combo)

        self.label_edit = QLineEdit()
        form.addRow("Étiquette :", self.label_edit)

        self.color_btn = QPushButton()
        self.color_btn.setFixedHeight(28)
        self._update_color_btn()
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow("Couleur :", self.color_btn)

        self.lw_spin = QDoubleSpinBox()
        self.lw_spin.setRange(0.5, 10.0)
        self.lw_spin.setSingleStep(0.5)
        self.lw_spin.setValue(1.5)
        form.addRow("Épaisseur :", self.lw_spin)

        self.ls_combo = QComboBox()
        for name, val in [("Continu", "-"), ("Tirets", "--"),
                          ("Pointillés", ":"), ("Tiret-point", "-."), ("Aucun", "None")]:
            self.ls_combo.addItem(name, val)
        form.addRow("Style ligne :", self.ls_combo)

        self.mk_combo = QComboBox()
        for name, val in [("Aucun", "None"), ("Point", "."), ("Cercle", "o"),
                          ("Carré", "s"), ("Triangle", "^"), ("Losange", "D"),
                          ("Croix", "x"), ("Plus", "+")]:
            self.mk_combo.addItem(name, val)
        form.addRow("Marqueur :", self.mk_combo)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("✔  Ajouter")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("✘  Annuler")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._on_file_changed(0)

    def _on_file_changed(self, _idx):
        df = self.file_combo.currentData()
        if df is None:
            return
        cols = df.col_names
        self.x_combo.clear()
        self.y_combo.clear()
        for c in cols:
            self.x_combo.addItem(c)
            self.y_combo.addItem(c)
        self.x_combo.setCurrentIndex(0)
        self.y_combo.setCurrentIndex(min(1, len(cols) - 1))
        self.label_edit.setText(f"{df.name} — {cols[min(1, len(cols)-1)]}")

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._chosen_color), self)
        if color.isValid():
            self._chosen_color = color.name()
            self._update_color_btn()

    def _update_color_btn(self):
        self.color_btn.setStyleSheet(
            f"background-color: {self._chosen_color}; border: 1px solid #555;"
        )
        self.color_btn.setText(self._chosen_color)

    def get_plot_entry(self) -> PlotEntry:
        df = self.file_combo.currentData()
        return PlotEntry(
            dat_file=df,
            x_col=self.x_combo.currentIndex(),
            y_col=self.y_combo.currentIndex(),
            color=self._chosen_color,
            label=self.label_edit.text() or df.name,
            linewidth=self.lw_spin.value(),
            linestyle=self.ls_combo.currentData(),
            marker=self.mk_combo.currentData(),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Widget d'une ligne dans la liste des tracés actifs
# ═════════════════════════════════════════════════════════════════════════════

class PlotItemWidget(QWidget):
    def __init__(self, entry: PlotEntry, on_toggle, on_remove, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        dot = QLabel()
        dot.setFixedSize(16, 16)
        dot.setStyleSheet(f"background-color:{entry.color}; border-radius:3px;")
        layout.addWidget(dot)

        chk = QCheckBox()
        chk.setChecked(entry.visible)
        chk.toggled.connect(lambda v: on_toggle(entry, v))
        layout.addWidget(chk)

        lbl = QLabel(entry.label)
        lbl.setFont(QFont("Courier New", 9))
        layout.addWidget(lbl, stretch=1)

        rm = QPushButton("✕")
        rm.setFixedSize(22, 22)
        rm.setStyleSheet("color:#c0392b; font-weight:bold; border:none;")
        rm.setToolTip("Supprimer ce tracé")
        rm.clicked.connect(lambda: on_remove(entry))
        layout.addWidget(rm)


# ═════════════════════════════════════════════════════════════════════════════
#  ██████╗  █████╗ ████████╗██╗   ██╗██╗███████╗██╗    ██╗███████╗██████╗
#  ██╔══██╗██╔══██╗╚══██╔══╝██║   ██║██║██╔════╝██║    ██║██╔════╝██╔══██╗
#  ██║  ██║███████║   ██║   ██║   ██║██║█████╗  ██║ █╗ ██║█████╗  ██████╔╝
#  ██║  ██║██╔══██║   ██║   ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║██╔══╝  ██╔══██╗
#  ██████╔╝██║  ██║   ██║    ╚████╔╝ ██║███████╗╚███╔███╔╝███████╗██║  ██║
#  ╚═════╝ ╚═╝  ╚═╝   ╚═╝     ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
#
#  Classe centrale — fonctionne seule OU intégrée dans un autre layout
# ═════════════════════════════════════════════════════════════════════════════

class DatViewerWidget(QWidget):
    """
    Widget autonome et réutilisable pour visualiser des fichiers .dat.

    Signaux
    -------
    status_message(str)
        Émis à chaque changement d'état (chargement, ajout/suppression de
        tracé…). Branchez-le sur votre propre QStatusBar si besoin :
            widget.status_message.connect(self.statusBar().showMessage)
        Quand il n'est pas branché, le message s'affiche dans la barre
        de statut interne du widget.

    API publique
    ------------
    load_files(paths: list[str])   — charger des fichiers par code
    clear_all_plots()              — supprimer tous les tracés
    dat_files                      — liste des DatFile chargés (lecture)
    plot_entries                   — liste des PlotEntry actifs (lecture)
    """

    # Signal émis à chaque message de statut
    status_message = pyqtSignal(str)

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

        self.dat_files: list[DatFile] = []
        self.plot_entries: list[PlotEntry] = []
        self._color_index = 0
        self._grid_on = True
        self._legend_on = True

        self._build_ui()

        if apply_stylesheet:
            self.setStyleSheet(STYLESHEET)
            self._apply_mpl_theme()

        self._post_status("Prêt — Chargez des fichiers .dat pour commencer.")

    # ── Construction UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Barre d'outils interne (QToolBar utilisée comme widget) ──────────
        # Fonctionne que le widget soit dans un QMainWindow ou non.
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setMovable(False)

        act_open = QAction("📂  Charger .dat", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_load_btn)
        toolbar.addAction(act_open)

        toolbar.addSeparator()

        act_add = QAction("+  Ajouter un tracé", self)
        act_add.setShortcut("Ctrl+P")
        act_add.triggered.connect(self.add_plot)
        toolbar.addAction(act_add)

        act_clear = QAction("🗑  Effacer tracés", self)
        act_clear.triggered.connect(self.clear_all_plots)
        toolbar.addAction(act_clear)

        toolbar.addSeparator()

        act_grid = QAction("⊞  Grille", self)
        act_grid.setCheckable(True)
        act_grid.setChecked(True)
        act_grid.toggled.connect(self._toggle_grid)
        toolbar.addAction(act_grid)

        act_legend = QAction("📋  Légende", self)
        act_legend.setCheckable(True)
        act_legend.setChecked(True)
        act_legend.toggled.connect(self._toggle_legend)
        toolbar.addAction(act_legend)

        root.addWidget(toolbar)

        # ── Contenu principal ────────────────────────────────────────────────
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(main_splitter, stretch=1)

        # Panneau gauche
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        ll.setSpacing(6)

        fg = QGroupBox("📁 Fichiers chargés")
        fl = QVBoxLayout(fg)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        fl.addWidget(self.file_list)
        fbr = QHBoxLayout()
        load_btn = QPushButton("📂 Charger")
        load_btn.clicked.connect(self._on_load_btn)
        rm_btn = QPushButton("🗑 Retirer")
        rm_btn.clicked.connect(self.remove_selected_file)
        fbr.addWidget(load_btn)
        fbr.addWidget(rm_btn)
        fl.addLayout(fbr)
        ll.addWidget(fg, stretch=2)

        ig = QGroupBox("ℹ Informations")
        il = QVBoxLayout(ig)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Courier New", 9))
        il.addWidget(self.info_text)
        ll.addWidget(ig, stretch=3)

        main_splitter.addWidget(left)

        # Panneau droit
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSpacing(4)

        right_split = QSplitter(Qt.Orientation.Vertical)

        plot_w = QWidget()
        plot_l = QVBoxLayout(plot_w)
        plot_l.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(9, 5), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        plot_l.addWidget(self.nav_toolbar)
        plot_l.addWidget(self.canvas)
        right_split.addWidget(plot_w)

        pg = QGroupBox("📈 Tracés actifs")
        pl = QVBoxLayout(pg)
        pbr = QHBoxLayout()
        add_btn = QPushButton("+  Ajouter un tracé")
        add_btn.clicked.connect(self.add_plot)
        clr_btn = QPushButton("🗑  Tout effacer")
        clr_btn.clicked.connect(self.clear_all_plots)
        pbr.addWidget(add_btn)
        pbr.addWidget(clr_btn)
        pbr.addStretch()
        pl.addLayout(pbr)
        self.plots_list_widget = QListWidget()
        self.plots_list_widget.setSpacing(2)
        pl.addWidget(self.plots_list_widget)
        right_split.addWidget(pg)
        right_split.setSizes([520, 200])

        rl.addWidget(right_split)
        main_splitter.addWidget(right)
        main_splitter.setSizes([300, 980])

        # ── Barre de statut interne ──────────────────────────────────────────
        # Visible uniquement quand aucun parent n'a branché le signal
        # status_message sur sa propre barre. On la cache dès qu'un slot
        # externe est connecté (voir _post_status).
        self._status_label = QLabel("Prêt")
        self._status_label.setObjectName("status_label")
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        root.addWidget(self._status_label)

        # Connecter le signal sur la barre interne par défaut
        self.status_message.connect(self._status_label.setText)

        self._refresh_plot()

    # ── Thème matplotlib ─────────────────────────────────────────────────────

    def _apply_mpl_theme(self):
        self.figure.patch.set_facecolor("#21252b")
        self.ax.set_facecolor("#282c34")
        for spine in self.ax.spines.values():
            spine.set_color("#3e4451")
        self.ax.tick_params(colors="#abb2bf")
        self.ax.xaxis.label.set_color("#abb2bf")
        self.ax.yaxis.label.set_color("#abb2bf")
        self.ax.title.set_color("#61afef")

    # ── Statut ───────────────────────────────────────────────────────────────

    def _post_status(self, msg: str):
        """Émet le signal de statut (vers barre interne et/ou barre parent)."""
        self.status_message.emit(msg)

    def hide_internal_statusbar(self):
        """
        Appelez cette méthode si vous branchez status_message sur une barre
        externe et que vous souhaitez masquer la barre interne.
        """
        self._status_label.hide()

    # ── Chargement de fichiers ───────────────────────────────────────────────

    def _on_load_btn(self):
        """Ouvre la boîte de dialogue de sélection de fichiers."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Charger des fichiers .dat",
            "", "Fichiers DAT (*.dat);;Tous les fichiers (*)"
        )
        if paths:
            self.load_files(paths)

    def load_files(self, paths: list[str]):
        """
        Charge une liste de fichiers .dat.
        Peut être appelé par code depuis un script parent.

        Parameters
        ----------
        paths : list[str]
            Chemins absolus ou relatifs vers les fichiers .dat.
        """
        loaded = 0
        for path in paths:
            path = os.path.abspath(path)
            if any(df.path == path for df in self.dat_files):
                continue
            try:
                df = DatFile(path)
                self.dat_files.append(df)
                item = QListWidgetItem(f"  {df.name}")
                item.setData(Qt.ItemDataRole.UserRole, df)
                self.file_list.addItem(item)
                loaded += 1
            except Exception as e:
                QMessageBox.warning(
                    self, "Erreur",
                    f"Impossible de charger :\n{path}\n\n{e}"
                )
        if loaded:
            self._post_status(
                f"{loaded} fichier(s) chargé(s). Total : {len(self.dat_files)}."
            )
            self.file_list.setCurrentRow(self.file_list.count() - 1)

    def remove_selected_file(self):
        """Retire le fichier sélectionné et ses tracés associés."""
        item = self.file_list.currentItem()
        if item is None:
            return
        df = item.data(Qt.ItemDataRole.UserRole)
        for entry in [e for e in self.plot_entries if e.dat_file is df]:
            self._remove_plot_entry(entry)
        self.dat_files.remove(df)
        self.file_list.takeItem(self.file_list.row(item))
        self.info_text.clear()
        self._post_status(f"Fichier retiré : {df.name}")

    def _on_file_selected(self, current, _previous):
        if current is None:
            self.info_text.clear()
            return
        df = current.data(Qt.ItemDataRole.UserRole)
        self.info_text.setText(df.info_text())

    # ── Gestion des tracés ───────────────────────────────────────────────────

    def add_plot(self):
        """Ouvre la boîte de dialogue d'ajout de tracé."""
        if not self.dat_files:
            QMessageBox.information(
                self, "Aucun fichier",
                "Chargez d'abord au moins un fichier .dat."
            )
            return
        dlg = AddPlotDialog(self.dat_files, self._color_index, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            entry = dlg.get_plot_entry()
            self._color_index += 1
            self.plot_entries.append(entry)
            self._add_plot_item_widget(entry)
            self._refresh_plot()
            self._post_status(f"Tracé ajouté : {entry.label}")

    def _add_plot_item_widget(self, entry: PlotEntry):
        item = QListWidgetItem(self.plots_list_widget)
        item.setData(Qt.ItemDataRole.UserRole, entry.id)
        w = PlotItemWidget(entry, self._on_plot_toggle, self._remove_plot_entry)
        item.setSizeHint(w.sizeHint())
        self.plots_list_widget.addItem(item)
        self.plots_list_widget.setItemWidget(item, w)

    def _on_plot_toggle(self, entry: PlotEntry, visible: bool):
        entry.visible = visible
        self._refresh_plot()

    def _remove_plot_entry(self, entry: PlotEntry):
        for i in range(self.plots_list_widget.count()):
            item = self.plots_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == entry.id:
                self.plots_list_widget.takeItem(i)
                break
        self.plot_entries.remove(entry)
        self._refresh_plot()
        self._post_status(f"Tracé supprimé : {entry.label}")

    def clear_all_plots(self):
        """Supprime tous les tracés actifs."""
        self.plot_entries.clear()
        self.plots_list_widget.clear()
        self._color_index = 0
        self._refresh_plot()
        self._post_status("Tous les tracés ont été supprimés.")

    # ── Rendu matplotlib ─────────────────────────────────────────────────────

    def _refresh_plot(self):
        self.ax.cla()
        self.ax.set_facecolor("#282c34")
        for spine in self.ax.spines.values():
            spine.set_color("#3e4451")
        self.ax.tick_params(colors="#abb2bf")
        self.ax.xaxis.label.set_color("#abb2bf")
        self.ax.yaxis.label.set_color("#abb2bf")
        self.ax.title.set_color("#61afef")

        if self._grid_on:
            self.ax.grid(True, color="#3e4451", linestyle="--", linewidth=0.7, alpha=0.8)

        has_data = False
        for entry in self.plot_entries:
            if not entry.visible:
                continue
            df = entry.dat_file
            if df.data is None or df.n_cols <= entry.y_col:
                continue
            x = df.data[:, entry.x_col]
            y = df.data[:, entry.y_col]
            ls = entry.linestyle if entry.linestyle != "None" else ""
            mk = entry.marker if entry.marker != "None" else ""
            line, = self.ax.plot(
                x, y,
                color=entry.color,
                label=entry.label,
                linewidth=entry.linewidth,
                linestyle=ls or "-" if not mk else ("" if not ls else ls),
                marker=mk or None,
                markersize=4,
            )
            entry.line_object = line
            has_data = True

        if has_data and self._legend_on:
            self.ax.legend(
                facecolor="#2c313a", edgecolor="#3e4451",
                labelcolor="#abb2bf", fontsize=9
            )

        title = (
            "Aucun tracé — Ajoutez un tracé via le bouton +"
            if not self.plot_entries
            else f"{sum(1 for e in self.plot_entries if e.visible)} courbe(s) affichée(s)"
        )
        color = "#5c6370" if not self.plot_entries else "#61afef"
        self.ax.set_title(title, color=color)
        self.canvas.draw()

    def _toggle_grid(self, checked: bool):
        self._grid_on = checked
        self._refresh_plot()

    def _toggle_legend(self, checked: bool):
        self._legend_on = checked
        self._refresh_plot()


# ═════════════════════════════════════════════════════════════════════════════
#  Enveloppe QMainWindow — mode standalone uniquement
# ═════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    Fenêtre principale pour le mode standalone.
    Enveloppe DatViewerWidget et connecte son signal status_message
    à la QStatusBar native de QMainWindow.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAT File Viewer")
        self.resize(1280, 780)

        # Créer le widget central
        self.viewer = DatViewerWidget(parent=self, apply_stylesheet=True)

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

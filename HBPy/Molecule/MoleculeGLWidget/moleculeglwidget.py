from PyQt6.QtCore import (Qt, QPoint, pyqtSignal)  # NEW: pyqtSignal
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import (glClearColor,glEnable,glDisable, GL_DEPTH_TEST,GL_LIGHTING,
                       GL_LIGHT0,GL_COLOR_MATERIAL,glColorMaterial,
                       GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,glShadeModel,
                       GL_SMOOTH,glLightfv,GL_POSITION,GL_DIFFUSE,GL_SPECULAR,
                       glViewport,glMatrixMode,GL_PROJECTION,glLoadIdentity,
                       GL_MODELVIEW,glClear,GL_COLOR_BUFFER_BIT,GL_DEPTH_BUFFER_BIT,
                       glTranslatef,glRotatef,glPushMatrix,glColor3f,glPopMatrix,
                       glGetDoublev, glGetIntegerv,
                       GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT,
                       glBegin, glEnd, glVertex3f, GL_LINES,glLineWidth)
from OpenGL.GLU import (gluNewQuadric,gluQuadricNormals,GLU_SMOOTH,gluPerspective,
                        gluSphere,gluProject)
import math  # NEW
import numpy
import sys
#sys.path.append('/home/bulou/src/lib/site-packages/')
from HBPy.Molecule.Atom import Atom,COV_RADIUS,CPK_COLOR
from HBPy.Molecule.Crystal import Crystal






# objet "Molecule"
class MoleculeGLWidget(QOpenGLWidget):
    atomSelected = pyqtSignal(int)  # NEW: émis à chaque sélection (-1 si rien)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_atom = -1      # NEW: aucun atome sélectionné au départ
        self.atoms=[]
        self.bonds=[]
        self.molecule=None
        # Caméra
        #self.rot_x = -20.0
        #self.rot_y = 30.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.distance = 3.0
        self.center=[0.0,0.0,0.0]   # point de centrage
        glRotatef(180, 1.0, 0.0, 0.0)
        #glRotatef(self.rot_y, 0.0, 1.0, 0.0)

        
        # Pan (translation) -------------  # NEW
        self.pan_x = 0.0                   # NEW
        self.pan_y = 0.0                   # NEW
        self._pixel_to_world = 0.01        # NEW (mis à jour selon FOV/viewport)

        self._last_pos = QPoint()
        self._quadric = None

        # Projection
        self._fov_deg = 45.0               # NEW


    def set_molecule(self, mol):
        #print("mol")
        self.molecule=mol
        self.atoms=mol.atoms
        if self.molecule is not None:
            self.distance=2*numpy.linalg.norm(self.molecule.qmax-self.molecule.qmin)
        else:
            self.distance=1.0
        self.update()               # 2) on demande un repaint → paintGL() sera rappelé
    def initializeGL(self):
        glClearColor(0.08, 0.08, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_POSITION, (3.0, 4.0, 5.0, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.9, 0.9, 0.9, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.6, 0.6, 0.6, 1.0))

        self._quadric = gluNewQuadric()
        gluQuadricNormals(self._quadric, GLU_SMOOTH)

    # appelé à chaque redimensionnement (tu y ajustes la projection, viewport).
    def resizeGL(self, w, h):
        h = max(h, 1)
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self._fov_deg, w / float(h), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        self._update_pixel_scale()          # NEW
    def get_axis_vectors(self):
        """
        Retourne les 3 vecteurs (x_axis, y_axis, z_axis) correspondant
        aux axes de la molécule après application de la caméra
        (rot_x, rot_y, pan, distance).
        """
        # S'assurer que le contexte est courant
        self.makeCurrent()

        # Reconstituer les matrices comme dans paintGL()
        self._setup_gl_matrices_like_paint()   # tu l'as déjà définie

        # Récupérer la matrice MODELVIEW
        mv = glGetDoublev(GL_MODELVIEW_MATRIX)

        # OpenGL stocke en column-major : on reshape en conséquence
        m = numpy.array(mv, dtype=float).reshape((4, 4), order='F')

        # Les 3 premières colonnes (partie 3x3) sont les axes
        x_axis = m[0:3, 0]   # direction de l’axe X
        y_axis = m[0:3, 1]   # direction de l’axe Y
        z_axis = m[0:3, 2]   # direction de l’axe Z

        return x_axis, y_axis, z_axis
    def get_rotation_matrix(self):
        """
        Retourne la matrice de rotation 3×3 résultant des rotations
        appliquées par la souris (rot_x, rot_y).
        """
        # S'assurer que le contexte est courant
        self.makeCurrent()

        # Reconstruire les matrices comme dans paintGL()
        self._setup_gl_matrices_like_paint()

        # Récupération de la matrice MODELVIEW OpenGL
        mv = glGetDoublev(GL_MODELVIEW_MATRIX)

        # Conversion en matrice 4×4 en column-major (format OpenGL)
        M = numpy.array(mv, dtype=float).reshape((4, 4), order='F')

        # Extraction de la partie rotation (les 3x3 premières entrées)
        R = M[:3, :3].copy()

        return R
    def paintGL(self):
        """ pour dessiner la scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Caméra
        #1. reculer la camera à la distance self.distance
        glTranslatef(self.center[0], self.center[1], self.center[2]-self.distance)
        glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        glRotatef(self.rot_y, 0.0, 1.0, 0.0)

        # Appliquer le pan (y inversé pour correspondre au déplacement écran)  # NEW
        glTranslatef(self.pan_x, -self.pan_y, 0.0)                               # NEW

        if self.molecule is not None:
            axis_len = numpy.linalg.norm(self.molecule.qmax - self.molecule.qmin) * 1.1
        else:
            axis_len=1.0
        # === AXES XYZ ===
        glDisable(GL_LIGHTING)      # pour que les couleurs soient bien visibles
        glLineWidth(2.0)
        glBegin(GL_LINES)

        # Axe X (rouge)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_len, 0.0, 0.0)

        # Axe Y (vert)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_len, 0.0)

        # Axe Z (bleu)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_len)

        glEnd()
        glEnable(GL_LIGHTING)
        # === fin axes ===





        
        # Liaisons
        #glDisable(GL_LIGHTING)
        #glLineWidth(3.0)
        #glBegin(GL_LINES)
        #glColor3f(0.7, 0.7, 0.7)
        #for i, j in self.bonds:
        #    xi, yi, zi, *_ = self.atoms[i]
        #    xj, yj, zj, *_ = self.atoms[j]
        #    glVertex3f(xi, yi, zi)
        #    glVertex3f(xj, yj, zj)
        #glEnd()
        #glEnable(GL_LIGHTING)

        # Atomes
        #COV_RADIUS = {'H':0.20,'C':0.25,'N':0.25,'O':0.25,'Pt':0.35,'Pd':0.35,'Rh':0.33,'Ir':0.33,'Ru':0.33}
        #elt est le symbole de l’atome courant (ex. "O", "H", "Pt").
        #COV_RADIUS.get(elt, 0.28) demande :
        #« Donne-moi la valeur associée à la clé elt si elle existe dans le dictionnaire COV_RADIUS »
        #« Sinon, si elt n’existe pas, retourne 0.28 par défaut. »
        #for (x, y, z, r, color) in self.atoms:
        if self.molecule is not None:
        #if hasattr(self, "molecule"):   # pour verifier que l'attribu molecule est deja definit 
            for atm in self.molecule.atoms:
                r = COV_RADIUS.get(atm.elt, 0.28)
                color = CPK_COLOR.get(atm.elt, (0.6, 0.6, 0.6))
                glPushMatrix()
                glTranslatef(atm.q[0],atm.q[1],atm.q[2])
                glColor3f(*color)
                gluSphere(self._quadric, r, 32, 24)
                glPopMatrix()
    def _update_pixel_scale(self):          # NEW
        """Met à jour le facteur de conversion pixel→monde à la distance actuelle."""
        h = max(self.height(), 1)
        fov_rad = math.radians(self._fov_deg)
        # demi-hauteur du frustum à la distance courante :
        half_h_world = self.distance * math.tan(fov_rad / 2.0)
        # 1 pixel en unités monde (axe vertical). Horizontal identique si pixels carrés.
        self._pixel_to_world = (2.0 * half_h_world) / float(h)
        



            
    # ---- Interactions souris ----
    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):  # NEW
            self._last_pos = event.position().toPoint()
        # NEW: Ctrl + clic gauche = sélectionner un atome
        if (event.button() == Qt.MouseButton.RightButton and
            (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and
            hasattr(self, "molecule") and self.molecule is not None):
            self._show_atom_at(event.position().x(), event.position().y())
            
        if (event.button() == Qt.MouseButton.LeftButton and 
            (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and
            hasattr(self, "molecule") and self.molecule is not None):
            self._rm_atom_at(event.position().x(), event.position().y())


        # on laisse le comportement existant continuer (rotation/pan)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()

        if event.buttons() & Qt.MouseButton.LeftButton:
            # rotation
            self.rot_x += dy * 0.5
            self.rot_y += dx * 0.5
            self.update()

        if event.buttons() & Qt.MouseButton.MiddleButton:     # NEW (PAN)
            # déplacer en unités monde, proportionnel à la distance et au FOV
            self.pan_x += dx * self._pixel_to_world
            self.pan_y += dy * self._pixel_to_world
            self.update()

        self._last_pos = pos
        x_axis, y_axis, z_axis = self.get_axis_vectors()
        print("X axis:", x_axis)
        print("Y axis:", y_axis)
        print("Z axis:", z_axis)
        R = self.get_rotation_matrix()
        print(R)


    def wheelEvent(self, event):
        # Zoom
        delta = event.angleDelta().y() / 120.0
        self.distance = max(1.0, self.distance - 0.3 * delta)
        self._update_pixel_scale()          # NEW (le facteur dépend de la distance)
        self.update()
    # NEW
    def _setup_gl_matrices_like_paint(self):
        """Reconstruit les matrices de projection et de vue comme dans paintGL()."""
        # Projection identique à resizeGL()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        gluPerspective(self._fov_deg, w / float(h), 0.1, 100.0)

        # ModelView identique à paintGL()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -self.distance)
        glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        glRotatef(self.rot_y, 0.0, 1.0, 0.0)
        glTranslatef(self.pan_x, -self.pan_y, 0.0)

    # NEW
    def _rm_atom_at(self, x_f: float, y_f: float):
        """Sélectionne l’atome le plus proche du clic (x,y) en pixels, si dans un rayon."""
        if not hasattr(self, "molecule") or self.molecule is None:
            return

        # S'assurer que le contexte GL est actif ici
        self.makeCurrent()

        # 1) Refaire les mêmes matrices que pour le rendu
        self._setup_gl_matrices_like_paint()

        # 2) Récupérer matrices/viewport pour gluProject
        mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        pj = glGetDoublev(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)  # [x, y, width, height]

        # 3) Convertir chaque atome en coords écran et mesurer la distance au clic
        x = float(x_f)
        y = float(y_f)
        h = float(vp[3])
        best_i = -1
        best_d2 = float("inf")

        # Rayon de tolérance en pixels (ajuste si besoin)
        pick_radius_px = 14.0

        for i, atm in enumerate(self.molecule.atoms):
            wx, wy, wz = atm.q  # coordonnées monde
            sx, sy, sz = gluProject(wx, wy, wz, mv, pj, vp)  # coords fenêtre
            # Inverser Y pour comparer au QPoint (origine en haut)
            sy_screen = h - sy
            d2 = (sx - x)**2 + (sy_screen - y)**2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        # 4) Valider si dans le rayon de tolérance
        if best_i >= 0 and best_d2 <= pick_radius_px**2:
            if best_i != self.selected_atom:
                print(best_i)

                self.molecule.rm_atom(best_i)
                self.selected_atom = best_i
                self.atomSelected.emit(best_i)
                self.update()
        else:
            print("No atom selection")
            if self.selected_atom != -1:
                self.selected_atom = -1
                self.atomSelected.emit(-1)
                self.update()
    def _show_atom_at(self, x_f: float, y_f: float):
        """Sélectionne l’atome le plus proche du clic (x,y) en pixels, si dans un rayon."""
        if not hasattr(self, "molecule") or self.molecule is None:
            return

        # S'assurer que le contexte GL est actif ici
        self.makeCurrent()

        # 1) Refaire les mêmes matrices que pour le rendu
        self._setup_gl_matrices_like_paint()

        # 2) Récupérer matrices/viewport pour gluProject
        mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        pj = glGetDoublev(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)  # [x, y, width, height]

        # 3) Convertir chaque atome en coords écran et mesurer la distance au clic
        x = float(x_f)
        y = float(y_f)
        h = float(vp[3])
        best_i = -1
        best_d2 = float("inf")

        # Rayon de tolérance en pixels (ajuste si besoin)
        pick_radius_px = 14.0

        for i, atm in enumerate(self.molecule.atoms):
            wx, wy, wz = atm.q  # coordonnées monde
            sx, sy, sz = gluProject(wx, wy, wz, mv, pj, vp)  # coords fenêtre
            # Inverser Y pour comparer au QPoint (origine en haut)
            sy_screen = h - sy
            d2 = (sx - x)**2 + (sy_screen - y)**2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        # 4) Valider si dans le rayon de tolérance
        if best_i >= 0 and best_d2 <= pick_radius_px**2:
            if best_i != self.selected_atom:
                info=f'atom {best_i} : {self.molecule.atoms[best_i].idx} {self.molecule.atoms[best_i].elt}  {self.molecule.atoms[best_i].q}'
                print(f"moleculewidget.py> {info}")
                self.update()
        else:
            print("No atom selection")
            if self.selected_atom != -1:
                self.selected_atom = -1
                self.atomSelected.emit(-1)
                self.update()

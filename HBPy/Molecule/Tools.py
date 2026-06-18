from pathlib import Path
from datetime import datetime
import mimetypes
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import logging
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class FileInfo:
    """Classe pour regrouper les informations principales sur un fichier."""

    def __init__(self, filepath):
        self.path = Path(filepath)
        if not self.path.exists():
            raise FileNotFoundError(f"Le fichier '{filepath}' n'existe pas.")
        self.stat = self.path.stat()

    @property
    def name(self):
        return self.path.name

    @property
    def extension(self):
        return self.path.suffix

    @property
    def type_mime(self):
        return mimetypes.guess_type(self.path)[0] or "inconnu"

    @property
    def size(self):
        return self.stat.st_size  # en octets

    @property
    def modified(self):
        return datetime.fromtimestamp(self.stat.st_mtime)

    @property
    def created(self):
        return datetime.fromtimestamp(self.stat.st_ctime)

    @property
    def is_file(self):
        return self.path.is_file()

    @property
    def is_dir(self):
        return self.path.is_dir()

    def __repr__(self):
        return (f"<FileInfo {self.name} ({self.type_mime}, {self.size} octets)>")

    def as_dict(self):
        """Retourne les infos sous forme de dictionnaire."""
        return {
            "nom": self.name,
            "chemin": str(self.path.resolve()),
            "taille (octets)": self.size,
            "type": self.type_mime,
            "création": self.created.isoformat(),
            "modification": self.modified.isoformat(),
            "est_fichier": self.is_file,
            "est_dossier": self.is_dir,
        }



def get_peak_positions(z_coords,display=False,margin=1.0):
    # objectif : détecter les plans cristallographiques d'une nanoparticule
    #            calculer la distance interréticulaire moyenne selon l'axe $z$.
    # 'z_coords' est un array numpy contenant toutes les cotes z
    # Par exemple pour format XYZ standard
    # z_coords = np.loadtxt("data/xyz/NP_2050.xyz", skiprows=2, usecols=3)
    # La méthode gaussian_kde() transforme les coordonnées atomiques discrètes (z_coords)
    #   en une fonction de densité continue.
    # 2. Calculer le KDE (densité de probabilité)
    # C'est une méthode non-paramétrique qui permet d'estimer la Fonction de
    # Densité de Probabilité (PDF) d'une variable aléatoire.
    density = gaussian_kde(z_coords, bw_method=0.05) # Ajuster bw_method selon le bruit
    #margin = 1.0  # Marge en Angströms
    z_range = np.linspace(min(z_coords) - margin, max(z_coords) + margin, 1000)

    z_density = density(z_range)

        # 3. Trouver les pics
    #peaks, _ = find_peaks(z_density, height=np.max(z_density)*0.1)
    #peaks, properties = find_peaks(z_density)
    peaks, _ = find_peaks(z_density)
    #peaks, properties = find_peaks(z_density, height=np.max(z_density)*0.1)
    #logger.info(f"{properties}")
    # Seuil de détection : Le paramètre height=np.max(z_density)*0.1 permet de filtrer le bruit et de ne retenir que les
    #                      pics significatifs (ceux dépassant 10% du pic maximum).
    z_planes = z_range[peaks]


    plt.figure(figsize=(10, 6))
    plt.plot(z_range, z_density, label='Densité (KDE)', color='blue', lw=2)
    # Marque les pics détectés (les plans cristallins)
    plt.plot(z_planes, z_density[peaks], "x", color='red', label='Plans détectés', markersize=10)
    
    # Mise en forme scientifique
    plt.title(f"Détection des plans atomiques (HEA) \n {len(z_planes)} plans identifiés", fontsize=14)
    plt.xlabel("Position suivant l'axe z (Å)", fontsize=12)
    plt.ylabel("Densité de probabilité d'atomes", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("get_peak_positions.png",
                dpi=150,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1,
                facecolor='white')

    #plt.show()
    
    d_mean=0.0
    for i in range(len(z_planes)-1):
        d=z_planes[i+1]-z_planes[i]
        d_mean+=d
    d_mean=d_mean/(len(z_planes)-1)
    return z_planes,d_mean
#______________________________________________________________________________________________________
def mk_mean(series_list,expo=0):
#______________________________________________________________________________________________________
    """
    Méthode  : Interpolation sur une grille commune.
    
    Stratégie :
    1. Trouver la plage des abscisse (x) commune à toutes les séries
    2. Créer une grille uniforme sur cette plage
    3. Interpoler chaque série sur cette grille
    4. Moyenner
    
    Args:
        series_list: Liste de tuples (x, y)
    
    Returns:
        energy_common, intensity_mean, intensity_std
    """
    # Trouver la plage commune (intersection de toutes les séries)
    x_min = max(serie[0].min() for serie in series_list)
    x_max = min(serie[0].max() for serie in series_list)
    
    logger.info(f"Plage commune: [{x_min:.2f}, {x_max:.2f}]")
    
    # Créer une grille uniforme
    n_points = len(series_list[0][0])  # Utilise le nombre de points de la première série
    x_common = np.linspace(x_min, x_max, n_points)
    
    # Interpoler chaque série sur la grille commune
    interpolated_y = []
    
    for x,y in series_list:
        # Interpolation linéaire (ou 'cubic' pour plus de lissage)
        f = interp1d(x,y, kind='linear', fill_value='extrapolate')
        y_interp = f(x_common)
        interpolated_y.append(y_interp)
    
    # Convertir en array pour calculs vectorisés
    interpolated_y = np.array(interpolated_y)
    
    # Calculer moyenne et écart-type
    y_mean = np.mean(interpolated_y, axis=0)
    y_std  = np.std(interpolated_y, axis=0)
    
    return x_common, y_mean*x_common**expo, y_std


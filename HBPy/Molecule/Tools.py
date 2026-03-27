from pathlib import Path
from datetime import datetime
import mimetypes

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

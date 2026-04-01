from HBPy.Molecule.Crystal import Crystal
import pandas as pd


class MoleculeEntry:
    def __init__(self, mol_id, source,molecule):
        self.mol_id    = mol_id
        self.source    = source
        self.molecule  = molecule
        
        #self._molecules: dict[str, MoleculeEntry] = {}
        #self._active_id: str | None = None

class MoleculeSession:
    def __init__(self):
        self.molecules:list[MoleculeEntry] =[]

    def delete(self, mol_id: int):
        """Supprime la molécule ayant l'id donné et renumérote les suivantes."""
        self.molecules = [mol for mol in self.molecules if mol.mol_id != mol_id]
    
        # Renumérote les ids pour qu'ils restent continus (0, 1, 2...)
        for i, mol in enumerate(self.molecules):
            mol.mol_id = i
        
    def load(self,filename=""):
        id=len(self.molecules)
        molecule=Crystal()
        molecule.load_file(filename)
        molecule.MassCenter()
        molecule.get_element_distribution()
        molecule.get_structure()

        self.molecules.append(MoleculeEntry(id,filename,molecule))
        
    def add_molecule(self,source="",molecule=None):
        self.molecules.append(MoleculeEntry(len(self.molecules),source,molecule))


    def to_dataframe(self) -> pd.DataFrame:
        """Uniquement pour l'affichage dans QTableView."""
        return pd.DataFrame([
            {"id": mol.mol_id, "source": mol.source}
            for mol in self.molecules
        ])

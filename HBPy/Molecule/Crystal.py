import os
import numpy as np
import subprocess
import shutil



import HBPy
from HBPy.Molecule.Tools import FileInfo
from HBPy.Molecule.Atom import Atom

from mace.calculators import mace_mp
import ase
import ase.optimize

import sys
#sys.path.append('./lib/')
#import abtem

import abtem

import copy
import matplotlib.pyplot as plt
import pandas as pd
import random
from itertools import islice
#from mendeleev import element

from PIL import Image



#me=9.1093897e-31        /* electron mass */
ELECTRON=1.60919e-19
#ELECTRONSTAR 1.60919 /* unité réduite*/
NA=6.023e23
KB=8.617385e-5 #/* eV/K */
CONV=(NA*ELECTRON*1.0e-7)   #/* facteur conversion pour les forces */

import logging
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(funcName)s() - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
# ==========================================================================================
# CONSTANTES DE CONFIGURATION
# ==========================================================================================
class Config:
    """Constantes de configuration de l'application."""
    
    # Écran par défaut
    DEFAULT_SCREEN_INDEX = 0  # Écran secondaire si disponible
    # default seed
    SEED=0
        
# ==========================================================================================
class Crystal:
# ==========================================================================================
    #________________________________________________________________________________
    def __init__(self):
    #________________________________________________________________________________
        self.atoms=[]
        self.status = []
    #________________________________________________________________________________        
    def add_atom(self,elt='Au',q=[0.0,0.0,0.0]):
    #________________________________________________________________________________        
        self.atoms.append(Atom(elt=elt,q=q))
    #________________________________________________________________________________
    def from_ase_Atoms(self,atoms):
    #________________________________________________________________________________

        #for atm in self.atoms:
        #    atoms += ase.Atom(HBPy.Molecule.Atom.Z_from_elt[atm.elt],
                              #(atm.q[0],atm.q[1],atm.q[2]))
        for i,atome in enumerate(atoms):
            #logger.info(f"{self.atoms[i].elt} {type(self.atoms[i].q)} Atome {atome.symbol} en position {type(atome.position)}")
            self.atoms[i].q=atome.position
            #logger.info(f"{self.atoms[i].elt} {self.atoms[i].q} Atome {atome.symbol} en position {atome.position}")
        return atoms
    #________________________________________________________________________________
    #________________________________________________________________________________
    def abTEM(self,config,display=False):
    #________________________________________________________________________________
        output_dir = f"{config['root_dir']}/{config['train']['TEM_img_dir']}"
        logger.info(f"TEM images directory = {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        # Crée une boîte vide de 10x10x10 Å
        #cellsize=config['abtem']['cell scale']*2.0*max(self.qmax[0]-self.qmin[0],
        #                                               self.qmax[1]-self.qmin[1])
        cellsize=config['image']['xmax']-config['image']['xmin']
        logger.info(f"Cell size: {cellsize}")
        # -------------------------- ASE part ------------------------------------
        # pour l'instant on passe par ASE pour fournir la structure à abtem
        #import ase
        
        atoms=self.to_ase_Atoms(cell=[cellsize,cellsize,cellsize], pbc=True)

        #atoms = ase.Atoms(cell=[cellsize,cellsize,cellsize], pbc=True)
        #for atm in self.atoms:
        #    atoms += ase.Atom(HBPy.Molecule.Atom.Z_from_elt[atm.elt],
        #                      (atm.q[0],atm.q[1],atm.q[2]))
        atoms.center()
        if display:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            a=abtem.visualize.show_atoms(atoms,ax=ax1,
                                         title="Beam view", numbering=True, merge=False)
            a=abtem.visualize.show_atoms(atoms, ax=ax2,
                                         plane="xz",
                                         title="Side view", numbering=True,merge=False,
                                         legend=True)
            plt.show()
        logger.info(f"slice_thickness= {config['abtem']['dz']}  sampling={config['abtem']['dx']}")
        potential = abtem.Potential(atoms,
                                    slice_thickness= config['abtem']['dz'],
                                    sampling= config['abtem']['dx'])
        #print(dir(potential))
        logger.info(f"potential extansion: {potential.extent}")
        logger.info(f"potential origin: {potential.origin}")
        logger.info(f"potential shape: {potential.shape}")

        # fonction d'onde électronique qui est diffusée
        plane_wave = abtem.PlaneWave(energy = config['abtem']['energy']  )
        exit_wave = plane_wave.multislice(potential)
        # exécution du calcul
        exit_wave.compute()
        # Après avoir calculé l'onde de sortie, il faut lui appliquer les effets de l'optique
        # du microscope et la faire atteindre le plan du détecteur à l'aide d'une
        # fonction de transfert de modulation ( MTF ). 
        # Pour une simulation d'imagerie TEM , on utilise généralement une fonction de transfert
        # de contraste ( CTF ) pour la fonction de transfert de modulation ( FTM ) .
        # La CTF peut inclure des aberrations optiques aplanétiques telles que
        # * le défaut de mise au point,
        # * l'aberration sphérique,
        # * l'astigmatisme
        # * les aberrations d'ondes cohérentes d'ordre supérieur.
        # Elle peut également inclure des effets optiques plus complexes tels que
        # * la distorsion de champ,
        # * la rotation de l'image
        # * les aberrations planaires, où les aberrations varient en fonction de la
        # position.
    
        # Dans les expériences HRTEM réalistes, les fonctions d'onde doivent être amplifiées
        # par une lentille d'objectif, ce qui introduit des aberrations et élimine de fait
        # les grands angles de diffusion.
        # ici on applique un flou de 50 angstreom et une ouverture d'objectif de 20 mrad
        # exit_wave.apply_ctf(defocus=-30,
        #                    focal_spread=40,
        #                    semiangle_cutoff=20)#.intensity()#.show(cbar=True);
        ctf = abtem.CTF(defocus =config['abtem']['defocus'],
                        focal_spread =config['abtem']['focal spread'],
                        semiangle_cutoff=config['abtem']['semiangle cutoff'])
        image_wave = ctf.apply(exit_wave) 
        image = image_wave.intensity()

        logger.info(f"sampling={exit_wave.sampling}")
        logger.info(f"extent={exit_wave.extent}")
        logger.info(f"gpts={exit_wave.gpts}")
        fig, axes = plt.subplots(1,1, figsize=(10,10),
                                 gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
        a2=image.show(ax=axes)
        plt.axis("off")
        idx_img=0
        # Sauvegarde en PNG (ou autre format suivant l’extension)
        filename = os.path.join(output_dir,f"img_{idx_img:04d}.png")
        plt.savefig(filename,
                    dpi=150,
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=0.1,
                    facecolor='white')

        
    #________________________________________________________________________________        
    def build(self,elt='Pt',a=3.92,Nx=-1,Ny=-1,Nz=-1,materials='bulk',radius=-1.0):
    #________________________________________________________________________________

        if Nx<0 and Ny<0 and Nz<0 and radius<0.0:
            logger.error(f"Error in build")
            exit()
        
        if radius>0.0:
            Nx=int(2*radius/a)+1
            Ny=Nx
            Nz=Nx
        mat=Crystal()
        for iz in range(Nz):
            z=iz*a
            for iy in range(Ny):
                y=iy*a
                for ix in range(Nx):
                    x=ix*a
                    idx=len(mat.atoms)
                    mat.atoms.append(Atom(elt=elt,q=np.array([x,y,z]),idx=idx)),
                    idx=len(mat.atoms)
                    mat.atoms.append(Atom(elt=elt,q=np.array([x+0.5*a,y+0.5*a,z]),idx=idx))
                    idx=len(mat.atoms)
                    mat.atoms.append(Atom(elt=elt,q=np.array([x+0.5*a,y,z+0.5*a]),idx=idx))
                    idx=len(mat.atoms)
                    mat.atoms.append(Atom(elt=elt,q=np.array([x,y+0.5*a,z+0.5*a]),idx=idx))
        #mat.MassCenter()

        mat.get_structure()
        mat.status = [True]*len(mat.atoms)

        if materials=='NP':
            for i,atm in enumerate(mat.atoms):
                d=atm.distance_from_(mat.MC)
                if d<=radius:
                    idx=len(self.atoms)
                    self.atoms.append(Atom(elt=mat.atoms[i].elt,
                                           q=mat.atoms[i].q,
                                           idx=idx)),
        self.status = [True]*len(self.atoms)
                    
    #________________________________________________________________________________        
    def core_shell(self,composition):
    #________________________________________________________________________________        
        self.get_element_distribution()
        self.MassCenter()
        self.origin_at_mass_center()
        self.get_structure()
        for atm in self.atoms:
            d=atm.distance_from_(self.MC)
            for rmax, elt in composition:
                if d<=rmax:
                    atm.elt=elt
                    break
        self.get_element_distribution()


    #________________________________________________________________________________        
    def duplicate(self):
    #________________________________________________________________________________        
        new_crystal=Crystal()
        for i in range(len(self.atoms)):
            new_crystal.atoms.append(self.atoms[i].duplicate())
        new_crystal.status = [True]*len(new_crystal.atoms)
        return new_crystal
    
    #________________________________________________________________________________        
    def reindex(self):
    #________________________________________________________________________________        
        i=0
        for atm in self.atoms:
            atm.idx=i
            i+=1
            
    #________________________________________________________________________________        
    def file_info(self,filename):
    #________________________________________________________________________________        

        self.filenfo = FileInfo(filename)
        print(f"# Crystal.py > loading {filename}")
        if filename.split(".")[-1] == "xyz":
            self.filenfo.nstruct=1
            
            with open(filename, 'r') as f:
                # 1. Lire le nombre d'atomes (ligne 0)
                line0 = f.readline()
                if not line0: return
                self.natoms = int(line0.strip())
                
                # 2. Sauter la ligne de commentaire (ligne 1)
                next(f)
    
                # 3. Lire EXACTEMENT self.natoms lignes après les deux premières
                # islice(itérable, start, stop)
                self.atoms = []
                for i, line in enumerate(islice(f, 0, self.natoms)):
                    parts = line.split()
                    if not parts: continue # Sécurité ligne vide
        
                    self.atoms.append(
                        Atom(
                            elt=parts[0],
                            q=np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                            idx=i
                        )
                    )
            self.reindex()
            self.status = [True]*len(self.atoms)
        else :
            print(f"Only simple xyz files can be read!")
                
            # i=0
            # self.filenfo.nstruct=0
            # struct=[]
        
            # while i<len(data):
            #     if len(data[i].split()) == 1:
            #         atoms=[]
            #         self.filenfo.nstruct=self.filenfo.nstruct+1
            #         natom=int(data[i]) ; i=i+1
            #         i=i+1
            #         for j in range(natom):
            #             line=data[i].split()
            #             idx=len(self.atoms)
            #             atoms.append(
            #                 Atom(
            #                     elt=line[0],
            #                     q=np.array([float(line[1]),
            #                                 float(line[2]),
            #                                 float(line[3])]),
            #                     idx=idx))
                        
            #             i=i+1
            #         struct.append(atoms)
            #         del atoms
            # print(filename,len(data),len(struct)," structure(s)")
            # self.atoms=struct[-1]
            # self.reindex()
            # self.status = [True]*len(self.atoms)
        
    #________________________________________________________________________________        
    def load_file(self,filename):
    #________________________________________________________________________________        
        self.file_info(filename)
        self.MassCenter()
        self.status = [True]*len(self.atoms)
        self.update_distances()
        self.get_element_distribution()
        self.get_structure()
    #________________________________________________________________________________        
    def mixing(self,nexchange: int=1,seed: int=Config.SEED):
    #________________________________________________________________________________        
        #random.seed(seed)

        for i in range(nexchange):
            self.get_element_distribution()
            # 1. On tire 2 index au hasard parmi les éléments disponibles
            idx=random.sample(range(len(self.list_elt)), 2)
            # 2. On récupère les valeurs correspondantes
            elts = [self.list_elt[i] for i in idx]
            logger.info(f"Index : {idx}")
            logger.info(f"Valeurs : {elts}")
            list_idx_exc=[]
            for elt in elts:
                #list_exc.append(self.pos_elt[random.randint(0,len(self.pos_elt[elt]))])
                idx=random.randrange(len(self.pos_elt[elt]))
                list_idx_exc.append(self.pos_elt[elt][idx])
                logger.info(f"{len(self.pos_elt[elt])} {idx} {list_idx_exc[-1]}")
                logger.info(f"{elt} {self.pos_elt[elt]} ")
            logger.info(f"{self.atoms[list_idx_exc[0]].elt} -> {elts[1]}")
            logger.info(f"{self.atoms[list_idx_exc[1]].elt} -> {elts[0]}")
            self.atoms[list_idx_exc[0]].elt=elts[1]
            self.atoms[list_idx_exc[1]].elt=elts[0]
        self.get_element_distribution()
    #________________________________________________________________________________        
    def exchange(self):
    #________________________________________________________________________________        
        self.get_element_distribution()
        
        elt1=self.atoms[0].elt
        if self.list_elt[0]==elt1:
            elt2=self.list_elt[1]
        else:
            elt2=self.list_elt[0]
            
        if len(self.list_elt)==2:
            for atm in self.atoms:
                if atm.elt==elt1:
                    atm.elt='X'
                else:
                    atm.elt=elt1
            for atm in self.atoms:
                if atm.elt=='X':
                    atm.elt=elt2
        self.get_element_distribution()
            
        print(f"{self.list_elt}")
            
    #________________________________________________________________________________        
    def rm_atom(self,idx=-1):
    #________________________________________________________________________________        
        if idx>=0:
            del self.atoms[idx]
            print("removing ",idx)

    #________________________________________________________________________________        
    def energy(self,FF,callback=None):
    #________________________________________________________________________________        
        for atm in self.atoms:
            atm.Erep=0.0
            atm.Eattsqr=0.0
        for atmi in self.atoms:
            elti=atmi.elt
            for atmj in self.atoms:
                if atmj.idx>atmi.idx:
                    eltj=atmj.elt
                    posj_in_atmi=atmi.idx_neigh.index(atmj.idx)
                    posi_in_atmj=atmj.idx_neigh.index(atmi.idx)
                    rij=atmi.d[posj_in_atmi]
                    #R=atmj.R[posj_in_atmi])
                    A=FF.tbsma[(elti,eltj)].A
                    p=FF.tbsma[(elti,eltj)].p
                    xi=FF.tbsma[(elti,eltj)].xi
                    q=FF.tbsma[(elti,eltj)].q
                    r0=FF.tbsma[(elti,eltj)].r0
                    alpha=rij/r0-1.0
                    nrjrep=A*np.exp(-p*alpha)
                    nrjatt=xi*xi*np.exp(-2*q*alpha)
                    atmi.Erep    = atmi.Erep    + nrjrep
                    atmi.Eattsqr = atmi.Eattsqr + nrjatt
                    atmj.Erep    = atmj.Erep    + nrjrep
                    atmj.Eattsqr = atmj.Eattsqr + nrjatt


        self.Epot=0.0
        for atm in self.atoms:
            atm.Eb=np.sqrt(atm.Eattsqr)
            atm.Esite=atm.Erep-atm.Eb
            self.Epot=self.Epot+atm.Esite
            #print(atm.Esite)
            #print(atm.F)

    #________________________________________________________________________________        
    def FEFF_run(self,config):
    #________________________________________________________________________________        
        for pgm in config['list_pgm']:
            logger.info(f"{100*'#'}\n{pgm}")
            subprocess.run([config["feff_dir"]+"/"+pgm],
                           capture_output=False, 
                           text=True, 
                           check=True)

            
    #________________________________________________________________________________        
    def FEFF_create_input_file(self,
                               config,
                               absorber_idx:int,
                               T:float=300.0): 
    #________________________________________________________________________________        

        output_dir = f"{config['input_save_dir']}"
        logger.info(f"Input FEFF files directory = {output_dir}/{config['filename']}")
        
        
        # Positionner l'origine sur l'atome absorbeur
        tmp_molecule=self.duplicate()
        tmp_molecule.origin_at(origin=self.atoms[absorber_idx].q)
        absorber = tmp_molecule.atoms[absorber_idx]


        with open(config['filename'], "w") as f:
            f.write(f"TITLE {config['TITLE']}\n")
            #     # *Cu at 190K, Debye temp 315K (Ashcroft & Mermin)
            #     # DEBYE 190 315 0
            f.write(f"DEBYE {T} {config['DEBYE_TEMP']} 0\n")
            f.write(f"SCF {config['SCF_RADIUS']}\n")
            f.write(f"EXAFS {config['EXAFS']}\n")
            f.write(f"RPATH {config['RPATH']}\n")

            if absorber.elt not in config['EDGE']:
                logger.error(f"Edge of {absorber.elt} unknown!")
                exit()
            f.write(f"EDGE {config['EDGE'][absorber.elt]}\n")
            f.write(f"CONTROL\t1 1 1 1 1 1\n")            
            
            
            list_atm={}
            #     # Autres atomes
            for atm in tmp_molecule.atoms:
                if atm.idx != absorber.idx:
                    # Calculer la distance
                    R = atm.q - absorber.q
                    d = np.linalg.norm(R)
                    if d<config["RMAX"]: list_atm[atm.elt]=atm.idx


            # Section POTENTIALS
            f.write(f'\nPOTENTIALS\n')
            f.write(f' {0:>4d} {HBPy.Molecule.Atom.Z_from_elt[absorber.elt]:>5d} {absorber.elt:>7s}\n')
            #     # Liste des éléments uniques 
            for i, elt in enumerate(self.list_elt, start=1):
                if elt in list_atm.keys():
                    f.write(f' {i:>4d} {HBPy.Molecule.Atom.Z_from_elt[elt]:>5d} {elt:>7s}\n')
            # Section ATOMS
            f.write(f'\nATOMS\n')
            f.write(
                f' {absorber.q[0]:>10.6f} {absorber.q[1]:>10.6f} {absorber.q[2]:>10.6f} '
                f'{0:>4d} {absorber.elt:>5s} {0:>8.4f} (Absorbeur)\n'
                 )

            #     # Autres atomes
            for atm in tmp_molecule.atoms:
                if atm.idx != absorber.idx:
                    # Calculer la distance
                    R = atm.q - absorber.q
                    d = np.linalg.norm(R)
                    if d>config["RMAX"]: continue 
                    # Trouver l'indice du potentiel
                    ipot = self.list_elt.index(atm.elt) + 1
                    f.write(
                        f' {atm.q[0]:>10.6f} {atm.q[1]:>10.6f} {atm.q[2]:>10.6f} '
                        f'{ipot:>4d} {atm.elt:>5s} {d:>8.4f}\n'
                    )

            f.write(f'END\n')

        del tmp_molecule

    #________________________________________________________________________________        
    def force(self,idx_new,FF):
    #________________________________________________________________________________        
        for atm in self.atoms:
            atm.F[idx_new]=np.zeros(3)
        Ftot=np.zeros(3)
        for atmi in self.atoms:
            elti=atmi.elt
            Ebi=atmi.Eb
            for atmj in self.atoms:
                if atmj.idx>atmi.idx:
                    eltj=atmj.elt
                    posj_in_atmi=atmi.idx_neigh.index(atmj.idx)
                    posi_in_atmj=atmj.idx_neigh.index(atmi.idx)
                    rij=atmi.d[posj_in_atmi]
                    R=atmi.R[posj_in_atmi]
                    #R=atmj.R[posj_in_atmi])
                    A=FF.tbsma[(elti,eltj)].A
                    p=FF.tbsma[(elti,eltj)].p
                    xi=FF.tbsma[(elti,eltj)].xi
                    q=FF.tbsma[(elti,eltj)].q
                    r0=FF.tbsma[(elti,eltj)].r0
                    alpha=rij/r0-1.0
                    Pij=np.exp(-p*alpha)
                    Qij=np.exp(-2*q*alpha)
                    Ebj=atmj.Eb
                    if Ebi == 0 or Ebj == 0 or rij == 0 or r0 == 0:
                        print(f"⚠️ Problème : Ebi={Ebi}, Ebj={Ebj}, rij={rij}, r0={r0}")
                        continue
                    fac=-(2*A*p*Pij-(1.0/Ebi+1.0/Ebj)*xi*xi*q*Qij)/r0/rij
                    atmi.F[idx_new]=atmi.F[idx_new]+fac*R
                    atmj.F[idx_new]=atmj.F[idx_new]-fac*R
        #for atm in self.atoms:
        #    print(atm.F[idx_new],atm.F[(idx_new+1)%2])
        
    #________________________________________________________________________________        
    def optimize_ase(self,tol=1.0e-12,new_step=None):
    #________________________________________________________________________________

        model_path = '/home/bulou/.cache/mace/20231203mace128L1_epoch199model'
        
        calc = mace_mp(model=model_path,   # chemin explicite - model='medium',
                       device='cpu',
                       default_dtype='float32')
        # Charger la NP depuis le fichier XYZ sauvegardé à l'étape 1.2
        #atoms = ase.io.read('NP.xyz')
        atoms = self.to_ase_Atoms()
        
        
        logger.info(f"Structure chargée : {len(atoms)} atomes")
        logger.info(f"Composition : {atoms.get_chemical_formula()}")
        
        # Boîte de simulation avec vide autour de la NP (nécessaire pour MACE)
        atoms.center(vacuum=10.0)
        
        # Attacher le calculateur
        atoms.calc = calc
        
        # Energie avant minimisation
        e_avant = atoms.get_potential_energy()
        logger.info(f"Energie avant minimisation : {e_avant:.4f} eV")
        logger.info(f"Soit {e_avant/len(atoms):.4f} eV/atome")
        # Minimisation LBFGS
        logger.info("Démarrage de la minimisation...")
        traj_file = 'NP_minimisation.traj'
        opt = ase.optimize.LBFGS(atoms, trajectory=traj_file, logfile='minimisation.log')
        opt.run(fmax=0.05)   # convergence à 0.05 eV/Å sur les forces
        
        # Energie après minimisation
        e_apres = atoms.get_potential_energy()
        logger.info(f"Energie après minimisation  : {e_apres:.4f} eV")
        logger.info(f"Soit {e_apres/len(atoms):.4f} eV/atome")
        logger.info(f"Relaxation : {e_avant - e_apres:.4f} eV")
        
        # Sauvegarder la structure relaxée
        ase.io.write('NP_relaxed.xyz', atoms)
        logger.info("Structure relaxée sauvegardée dans NP_relaxed.xyz")
        
        self.from_ase_Atoms(atoms)
        self.origin_at_mass_center()


        
    #________________________________________________________________________________        
    def optimize(self,tol=1.0e-12,new_step=None):
    #________________________________________________________________________________        
        """
        new_step est une fonction callback, que l'on appel à chaque étape de l'optimisation
        pour suivre ou modifier l'exécution.
        Exemple :
        dans Py_ATOMOD, on suit graphiquement l''évolution de l'énergie du système durant
        l'optimisation, optimisation qui est réalisée en appelant la fonction optimize de Crystal.py.
        Pour faire, dans Py_ATOMOD, on utilise la commande
                self.molecule.optimize(new_step=self._on_opt_step,tol=tol)
        self.molecule est une instance Crystal qui possède la fonction optimize. A chaque nouvelle step, on appelle la
        fonction _on_opt_step de Py_ATOMOD qui se charge de mettre à jour la courbe nrj=f(step)
        """
        idx_t=0
        idx_newt=(idx_t+1)%2
        self.force(idx_t,self.FF)
        freq=10
        nstep=200*freq

        step=[]
        Ek=[]
        Epot=[]
        Etot=[]
        quench=True
        cvg=[0.0,0.0]
        dnrj=0.0
        
        dnrj=1.0
        istep=0
        while np.abs(dnrj)>tol:
            #for istep in range(nstep):
            self.move_atoms(idx_t)
            self.energy(self.FF)
            self.force(idx_newt,self.FF)
            self.update_p(idx_newt,quench=quench)
            Ek.append(self.Ek)
            Epot.append(self.Epot)
            Etot.append(self.Ek+self.Epot)
            step.append(istep)
            cvg[idx_newt]=Etot[-1]
            #if istep%freq == 0:
            #    print(Ek[-1],Epot[-1],Etot[-1])
            dnrj=cvg[idx_newt]-cvg[idx_t]

            if new_step is not None:
                try:
                    new_step(self, istep,step,Etot)
                except Exception as e:
                    print(f"[optimize on_step] {e}")
            print("%6d %12.6g %12.2e"%(istep,cvg[idx_newt],dnrj))
            idx_t=idx_newt
            idx_newt=(idx_t+1)%2
            istep=istep+1
        #plt.plot(Etot)
        #plt.show()
        self.save(prefix="last",fmt='xyz')

    
    #________________________________________________________________________________        
    def get_element_distribution(self):
    #________________________________________________________________________________        
        """
        Compter les occurrences de chaque type d'élément d'un Crystal déjà existant
        Input : liste des atomes constituant le Crystal.
        """
        from collections import Counter,defaultdict

        list_elt=[]
        for atm in self.atoms:
            list_elt.append(atm.elt)

        self.element_counts = Counter(list_elt)
        self.pos_elt={}
        self.list_elt=[]
        for elt, valeur in self.element_counts.items():
            self.pos_elt[elt]=[]
            self.list_elt.append(elt)
        for atm in self.atoms:
            self.pos_elt[atm.elt].append(atm.idx)

        self.nb_elt_differents = len(self.element_counts)
        #self.composition = len(self.element_counts)

    #________________________________________________________________________________        
    def get_structure(self):
    #________________________________________________________________________________        
        """
        pour récupérer divers informations sur la structure de l'objet Crystal
        """
        self.MassCenter()
        self.qmin=np.zeros(3)
        self.qmax=np.zeros(3)
        
        for atm in self.atoms:
            for i in range(3):
                if atm.q[i]<self.qmin[i]:
                    self.qmin[i]=atm.q[i]
                if atm.q[i]>self.qmax[i]:
                    self.qmax[i]=atm.q[i]

    #________________________________________________________________________________        
    def MassCenter(self,display=False):
    #________________________________________________________________________________        
        """ fonction calculant le centre de masse de la nanoparticule """
        self.MC=np.zeros(3)
        for i in range(len(self.atoms)):
            for k in range(3):
                self.MC[k]=self.MC[k]+self.atoms[i].q[k]
        for k in range(3):
            self.MC[k]=self.MC[k]/len(self.atoms)
        if display:
            logger.info(f"Mass center: {self.MC}")
    #________________________________________________________________________________        
    def move_atoms(self,idx_t,dt=1.0):
    #________________________________________________________________________________        
        for atm in self.atoms:
            atm.q=atm.q+CONV*dt*(atm.p+0.5*atm.F[idx_t]*dt)/atm.mass

        self.update_distances()
        
    #________________________________________________________________________________        
    def origin_at(self,origin=np.array([0.0,0.0,0.0])):
    #________________________________________________________________________________        
        for i in range(len(self.atoms)):
            for k in range(3):
                self.atoms[i].q[k]=self.atoms[i].q[k]-origin[k]
    #________________________________________________________________________________        
    def origin_at_mass_center(self):
    #________________________________________________________________________________        
        self.MassCenter()
        for i in range(len(self.atoms)):
            for k in range(3):
                self.atoms[i].q[k]=self.atoms[i].q[k]-self.MC[k]
        #self.MassCenter()
        self.get_structure()
    #________________________________________________________________________________        
    def save(self,prefix="crystal",fmt='xyz',directory='./'):
    #________________________________________________________________________________
        os.makedirs(directory, exist_ok=True)
        if fmt == 'xyz':
            f=open(f"{directory}/{prefix}.xyz",'w')
            f.write("%d\n\n"%(len(self.atoms)))
            for atom in self.atoms:
                f.write("%2s %12.6f %12.6f %12.6f\n"%(atom.elt,atom.q[0],atom.q[1],atom.q[2]))
            f.close()
        if fmt == 'xsf':
            f=open(f"{directory}/{prefix}.xsf",'w')
            f.write(" ANIMSTEPS        1\n")
            f.write(" CRYSTAL\n")
            f.write(" PRIMVEC           1\n")
            f.write(" %14.9f %14.9f %14.9f\n"%(self.L[0],0.0,0.0))
            f.write(" %14.9f %14.9f %14.9f\n"%(0.0,self.L[1],0.0))
            f.write(" %14.9f %14.9f %14.9f\n"%(0.0,0.0,self.L[2]))
            f.write("  CONVVEC           1\n")
            f.write(" %14.9f %14.9f %14.9f\n"%(self.L[0],0.0,0.0))
            f.write(" %14.9f %14.9f %14.9f\n"%(0.0,self.L[1],0.0))
            f.write(" %14.9f %14.9f %14.9f\n"%(0.0,0.0,self.L[2]))
            f.write("  PRIMCOORD           1\n")
            f.write("           %d           1\n"%(len(self.atoms)))
            for atom in self.atoms:
                f.write("%2s %12.6f %12.6f %12.6f\n"%(atom.elt,atom.q[0],atom.q[1],atom.q[2]))
        if fmt == 'lammps-data':
            f=open(f"{directory}/{prefix}.data",'w')
            f.write("\n");
            f.write("%d atoms\n"%len(self.atoms))
            f.write("1 atom types\n");
            f.write("%12.6f %12.6f xlo xhi\n"%(self.min[0],self.min[0]+self.L[0]))
            f.write("%12.6f %12.6f ylo yhi\n"%(self.min[1],self.min[1]+self.L[1]))
            f.write("%12.6f %12.6f zlo zhi\n"%(self.min[2],self.min[2]+self.L[2]))
            f.write("\n");
            f.write("Masses\n");
            f.write("\n");
            f.write("1 196.966552 # Au\n");
            f.write("\n");
            f.write("Atoms # atomic\n");
            f.write("\n");
            i=1
            itype=1
            for atom in self.atoms:
                f.write("%d %d %12.6f %12.6f %12.6f\n"%(i,itype,atom.q[0],atom.q[1],atom.q[2]))
                i=i+1

        f.close()

    #________________________________________________________________________________        
    def set_composition(self,composition):
    #________________________________________________________________________________        
        self.get_element_distribution()
        for elt in composition:
            if elt not in self.pos_elt:
                self.pos_elt[elt]=[]
                #logger.info(f"### {elt} {self.pos_elt[elt]} -> stoechiometry {len(self.pos_elt[elt])/len(self.atoms)}")

        seed=0 ; random.seed(seed)
        stoechiometry=1.0/len(composition)
        nmin=len(self.pos_elt[composition[0]])*stoechiometry
        idxfill=1
        while len(self.pos_elt[composition[0]])>nmin:
            # on choisit au hasard un des atomes de l'espèce en excés
            n = random.randrange(0, len(self.pos_elt[composition[0]]))   # 0 à 10 (11 exclu)

            if len(self.pos_elt[composition[idxfill]])>=nmin:
                idxfill=idxfill+1
            idx=self.pos_elt[composition[0]].pop(n)
            self.pos_elt[composition[idxfill]].append(idx)
            self.atoms[idx].elt=composition[idxfill]
        self.get_element_distribution()
        self.get_structure()





    #________________________________________________________________________________
    def to_ase_Atoms(self,cell=(0,0,0),pbc=False):
    #________________________________________________________________________________
        atoms = ase.Atoms(cell=cell, pbc=pbc)
        for atm in self.atoms:
            atoms += ase.Atom(HBPy.Molecule.Atom.Z_from_elt[atm.elt],
                              (atm.q[0],atm.q[1],atm.q[2]))
        return atoms
    #________________________________________________________________________________        
    def to_df(self):
    #________________________________________________________________________________        
        rows = []
        for atm in self.atoms:
            rows.append((atm.idx,atm.elt,atm.q[0],atm.q[1],atm.q[2]))
        df = pd.DataFrame(rows, columns=["idx","Element", "x", "y", "z"])
        return df
        
    #________________________________________________________________________________        
    def transform(self,radius=1.0,O=None):
    #________________________________________________________________________________        
        natom=0
        if O is None:
            O = self.MC
        for i in range(len(self.atoms)):
        #for atm,status in zip(self.atoms,self.save):
            atm=self.atoms[i]
            R=atm.q-O
            d = np.linalg.norm(R)
            if d<radius:
                natom=natom+1
                print(R,d)
                self.status[i]=True
            else:
                self.status[i]=False
        if natom == len(self.atoms):
            print("Bulk structure probably too small")
        new= copy.deepcopy(self)
        new.atoms= [atm for x,atm in zip(self.status,self.atoms) if x ==True]
        new.status = [True]*len(new.atoms)
        idx=0
        for atm in new.atoms:
            atm.idx=idx
            idx=idx+1
        new.update_distances()
        #print(len(new.atoms))
        
        #print([(x,atm.q) for x,atm in zip(self.save,self.atoms) if x ==True])
        return new
    #________________________________________________________________________________        
    def update_distances(self):
    #________________________________________________________________________________        
        for atmi in self.atoms:
            atmi.R=[]
            atmi.d=[]
            atmi.idx_neigh=[]
        for atmi in self.atoms:
            for atmj in self.atoms:
                if atmj.idx>atmi.idx:
                    R=atmj.q-atmi.q
                    d=np.sqrt(np.sum(R**2))
                    atmi.R.append(R)
                    atmi.d.append(d)
                    atmi.idx_neigh.append(atmj.idx)
                    atmj.R.append(-R)
                    atmj.d.append(d)
                    atmj.idx_neigh.append(atmi.idx)
            #print(atmi.R)
    #________________________________________________________________________________        
    def update_forces():
    #________________________________________________________________________________        
        pass
    #________________________________________________________________________________        
    def update_p(self,idx_new,dt=1.0,quench=False):
    #________________________________________________________________________________        
        self.Ek=0.0
        for atm in self.atoms:
            atm.p=atm.p+0.5*dt*(atm.F[(idx_new+1)%2]+atm.F[idx_new])
            if quench==True:
                if np.dot(atm.F[idx_new],atm.p) <= 0:
                    atm.p=np.zeros(3)
            #self.Ek=self.Ek+CONV*0.5*np.dot(atm.p,atm.p)/atm.mass
            self.Ek=self.Ek+CONV*0.5*np.linalg.norm(atm.p)**2/atm.mass
        self.T=2*self.Ek/(3*len(self.atoms)*KB)
    #________________________________________________________________________________
    def xyz2slice(self,config):
    #________________________________________________________________________________
        tmp=self.duplicate()
        tmp.origin_at(origin=np.array([self.qmin[0],self.qmin[1],self.qmin[2]]))
        tmp.get_structure()

        coords={}
        peak={}
        dpeak={}
        Npts={}
        d={}
        grid={
            'x':[],
            'y':[]
        }
        O={
            'x':0.0,
            'y':0.0,
            'z':0.0
        }
        
        #
        # on repère les plans atomiques de la nanoparticule dans les trois directions
        # de l'espace et on calcule la distance moyenne entre deux plans, dpeak
        #
        for i in ['x','y','z']:
            coords[i]=[] 
            peak[i]=[]
            dpeak[i]=[]
            Npts[i]=[]
            d[i]=[]
        for atm in tmp.atoms:
            for i,xyz in enumerate(['x','y','z']):
                coords[xyz].append(atm.q[i])
        for xyz in ['x','y','z']:
            peak[xyz],dpeak[xyz]=HBPy.Molecule.Tools.get_peak_positions(coords[xyz],display=False,margin=1.0)    
            logger.info(f"Number of plane(s) along {xyz}: {len(peak[xyz])}")
            logger.info(f"Mean interplane distancealong {xyz}: {dpeak[xyz]}")
            logger.info(f" {peak[xyz]}")

        # on calcule l'intervalle de discretisation d ainsi que le nombre de points pour
        # générer les maps selon les trois direction de l'espace
        for xyz in ['x','y','z']:
            d[xyz]=dpeak[xyz]/config['atomic presence probability map']['ninter'][xyz]
            Npts[xyz]=int(round(((len(peak[xyz])+2*config['nvaccum'])-1)*dpeak[xyz]/d[xyz]))
            logger.info(f"{xyz}: d={d[xyz]} Npts={Npts[xyz]}")
        
        for i,xyz in enumerate(['x','y','z']):
            O[xyz]=tmp.qmin[i]-config['nvaccum']*dpeak[xyz]
            grid[xyz]=np.linspace(O[xyz],
                                  tmp.qmax[i]+config['nvaccum']*dpeak[xyz],
                                  Npts[xyz])
            logger.info(f"{xyz} grid ({grid[xyz][0]}, {grid[xyz][-1]}) d={d[xyz]}")
            config['abtem'][xyz]=d[xyz]

        logger.info(f"{self.list_elt}")        
        volumes = {}  # dict: espèce -> volume 3D
        for sp in self.list_elt:
            volumes[sp] = np.zeros((Npts['x'], Npts['y'], Npts['z']), dtype=float)

        i_center={}
        i_min={}
        i_max={}
        nvxl={}
        subgrid={}
        localgrid={ }
        q={ }
        d2={}
        for xyz in ['x','y','z']:
            nvxl[xyz] = int(round((3 * config['atomic presence probability map']['sigma'] / d[xyz])))  # rayon en nombre de voxels
            logger.info(f"number of voxel: {nvxl[xyz]} rloc={nvxl[xyz]*d[xyz]/2}")
        for atom in tmp.atoms:
            sp = atom.elt
            vol = volumes[sp]
            #     # Indices du voisinage à affecter (±3 sigma)
            for i,xyz in enumerate(['x','y','z']):
                i_center[xyz] = int(round((atom.q[i] - O[xyz]) / d[xyz]))
                i_min[xyz] = np.clip(i_center[xyz] - nvxl[xyz], 0, Npts[xyz] - 1)
                i_max[xyz] = np.clip(i_center[xyz] + nvxl[xyz] + 1, 0, Npts[xyz])

                
                # Sous-grille locale
                subgrid[xyz] = grid[xyz][i_min[xyz]:i_max[xyz]]
            # La commande numpy.meshgrid sert à créer des grilles de coordonnées à partir de vecteurs
            # unidimensionnels. Elle transforme des listes de positions sur des axes (X, Y, Z...) en matrices
            # représentant toutes les combinaisons possibles de points dans l'espace.
            localgrid['x'], localgrid['y'],localgrid['z'] = np.meshgrid(subgrid['x'],subgrid['y'],subgrid['z'], indexing="ij")
            for i,xyz in enumerate(['x','y','z']):
                d2[xyz]=(localgrid[xyz]-atom.q[i])**2
            gauss = np.exp(-(d2['x']+d2['y']+d2['z']) / (2 * config['atomic presence probability map']['sigma']**2))
            vol[i_min['x']:i_max['x'], i_min['y']:i_max['y'], i_min['z']:i_max['z']] += gauss

        
        # sauvegarde des atomic presence probability maps
        output_dir = f"{config['root_dir']}/{config['train']['prob_maps_img_dir']}"

        logger.info(f"Prob_maps images directory = {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        list_filename={}
        for elt in config['structure']['composition']:
            list_filename[elt]=[]
        for sp in self.list_elt:
            vol=volumes[sp]
            # Optionnel : échelle globale fixe
            vmin = vol.min()
            vmax = vol.max()
            for k in range(Npts['z']):
                slice_z = vol[:, :, k]        # coupe dans le plan x-y
                fig, ax = plt.subplots(figsize=(6, 6))  # carré pour être sûr
                im = ax.imshow(
                    slice_z.T,
                    origin='lower',
                    extent=[grid['x'][0],grid['x'][-1],grid['y'][0],grid['y'][-1]],
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='nearest',
                    alpha=0.9
                )

                # impose ratio 1:1
                ax.set_aspect('equal')  # x et y même échelle

                # labels et titre
                z_val=grid['z'][0]+k*d['z']
                ax.set_title(f"Coupe à z = {z_val:.2f} Å  (k={k})")
                ax.set_xlabel("x (Å)")
                ax.set_ylabel("y (Å)")
            
                # *** SUPPRESSION DES ÉLÉMENTS GRAPHIQUES ***
                ax.set_xticks([])   # pas de ticks x
                ax.set_yticks([])   # pas de ticks y
                ax.set_xlabel("")   # pas de labels
                ax.set_ylabel("")
                ax.set_title("")    # pas de titre
                ax.axis('off')      # supprime l’axe et le cadre
            
                #fig.colorbar(im, ax=ax, label="densité")

                # sauvegarde {int(self.WD_lineedit_configidx.text()):04d}
                filename = os.path.join(output_dir, f"img_{0:04d}_{sp}_{k:04d}_{z_val:5.2f}.png")
                list_filename[sp].append(filename)
                plt.savefig(filename,
                            dpi=150,
                            bbox_inches='tight',
                            transparent=True,
                            pad_inches=0.1,
                            facecolor='white')

                #plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
        #logger.info(f"{grid['x'][0]},{grid['x'][-1]},{grid['y'][0]},{grid['y'][-1]}")
        config['image']['xmin']=grid['x'][0]
        config['image']['xmax']=grid['x'][-1]
        config['image']['ymin']=grid['y'][0]
        config['image']['ymax']=grid['y'][-1]
        # Version compacte
        
        #self.create_compact_summary(config)
        #for elt in config['structure']['composition']:
        #    logger.info(f"{list_filename[elt]}")

        nelt = len(config['structure']['composition'])
        ncol = max([len(list_filename[elt]) for elt in config['structure']['composition']])
        # Créer une grille d'images

        images={}
        for i, elt in enumerate(config['structure']['composition']):
            images[elt]=[]
            for j, filename in enumerate(list_filename[elt]):
                logger.info(f"{filename}")
                img = Image.open(filename)
                images[elt].append(img)
        img_w, img_h = images[config['structure']['composition'][0]][0].size
        canvas_w = ncol * img_w
        canvas_h = nelt * img_h
        canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
        for i, elt in enumerate(config['structure']['composition']):
            for j, filename in enumerate(list_filename[elt]):
                x = j * img_w
                y = i * img_h
                canvas.paste(images[elt][j], (x, y))

        canvas.save('summary.png', dpi=(300, 300))



import os
import numpy as np
import HBPy
from HBPy.Molecule.Tools import FileInfo, get_z_plane
from HBPy.Molecule.Atom import Atom

import sys
sys.path.append('./lib/')
import abtem


import copy
import matplotlib.pyplot as plt
import pandas as pd
import random
from itertools import islice
#from mendeleev import element

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
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        

class Crystal:
    def __init__(self):
        self.atoms=[]
        self.status = []


        
    def xyz2slice(self):
        tmp=self.duplicate()
  
        #tmp.origin_at(origin=np.array([-10.0,-10.0,-10.0]))
        tmp.get_structure()

        logger.info(f"Number of atoms = {len(tmp.atoms)}")
        z_coords=[]
        for atm in tmp.atoms:
            z_coords.append(atm.q[2])
        zp,dzmean=get_z_plane(z_coords)    
        logger.info(f"Number of plane(s): {len(zp)}")
        logger.info(f"Mean interplane distance: {dzmean}")
        logger.info(f" {zp}")
        dz=dzmean


        
    def abTEM(self,config,display=False):
        logger.info(f"TEM images directory = {config['train']['TEM_img_dir']}")
        output_dir = config['train']['TEM_img_dir']
        os.makedirs(output_dir, exist_ok=True)
        # Crée une boîte vide de 10x10x10 Å
        cellsize=config['abtem']['cell scale']*2.0*max(self.qmax[0]-self.qmin[0],
                                                       self.qmax[1]-self.qmin[1])
        # -------------------------- ASE part ------------------------------------
        # pour l'instant on passe par ASE pour fournir la structure à abtem
        import ase
        atoms = ase.Atoms(cell=[cellsize,cellsize,cellsize], pbc=True)
        for atm in self.atoms:
            atoms += ase.Atom(HBPy.Molecule.Atom.Z_from_elt[atm.elt],
                              (atm.q[0],atm.q[1],atm.q[2]))
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
        potential = abtem.Potential(atoms,
                                    slice_thickness= config['abtem']['dz'],
                                    sampling= config['abtem']['dx'])
        print(dir(potential))
        print(potential.extent)
        print(potential.origin)
        print(potential.shape)

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
        filename = os.path.join(config['train']['TEM_img_dir'],
                                f"img_{idx_img:04d}.png")
        plt.savefig(filename,
                    dpi=150,
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=0.1,
                    facecolor='white')

        
        
    def build(self,elt='Pt',a=3.92,Nx=-1,Ny=-1,Nz=-1,materials='bulk',radius=-1.0):

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
                    
    def core_shell(self,composition):
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


    def duplicate(self):
        new_crystal=Crystal()
        for i in range(len(self.atoms)):
            new_crystal.atoms.append(self.atoms[i].duplicate())
        new_crystal.status = [True]*len(new_crystal.atoms)
        return new_crystal
    
    def reindex(self):
        i=0
        for atm in self.atoms:
            atm.idx=i
            i+=1
            
    def file_info(self,filename):

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
        
    def load_file(self,filename):
        self.file_info(filename)
        self.MassCenter()
        self.status = [True]*len(self.atoms)
        self.update_distances()
        self.get_element_distribution()
        self.get_structure()
    def mixing(self,nexchange: int=1,seed: int=Config.SEED):
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
    def exchange(self):
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
            
    def add_atom(self,elt='Au',q=[0.0,0.0,0.0]):
        self.atoms.append(Atom(elt=elt,q=q))
    def rm_atom(self,idx=-1):
        if idx>=0:
            del self.atoms[idx]
            print("removing ",idx)

    def energy(self,FF,callback=None):
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
    def force(self,idx_new,FF):
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
    def optimize(self,tol=1.0e-12,new_step=None):
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

    def to_df(self):
        rows = []
        for atm in self.atoms:
            rows.append((atm.idx,atm.elt,atm.q[0],atm.q[1],atm.q[2]))
        df = pd.DataFrame(rows, columns=["idx","Element", "x", "y", "z"])
        return df
    
    def get_element_distribution(self):
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

    def get_structure(self):
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

    def MassCenter(self):
        """ fonction calculant le centre de masse de la nanoparticule """
        self.MC=np.zeros(3)
        for i in range(len(self.atoms)):
            for k in range(3):
                self.MC[k]=self.MC[k]+self.atoms[i].q[k]
        for k in range(3):
            self.MC[k]=self.MC[k]/len(self.atoms)
    def move_atoms(self,idx_t,dt=1.0):
        for atm in self.atoms:
            atm.q=atm.q+CONV*dt*(atm.p+0.5*atm.F[idx_t]*dt)/atm.mass

        self.update_distances()
        
    def origin_at(self,origin=np.array([0.0,0.0,0.0])):
        for i in range(len(self.atoms)):
            for k in range(3):
                self.atoms[i].q[k]=self.atoms[i].q[k]-origin[k]
    def origin_at_mass_center(self):
        self.MassCenter()
        for i in range(len(self.atoms)):
            for k in range(3):
                self.atoms[i].q[k]=self.atoms[i].q[k]-self.MC[k]
        #self.MassCenter()
        self.get_structure()
    def save(self,prefix="crystal",fmt='xyz'):
        if fmt == 'xyz':
            f=open(prefix+'.xyz','w')
            f.write("%d\n\n"%(len(self.atoms)))
            for atom in self.atoms:
                f.write("%2s %12.6f %12.6f %12.6f\n"%(atom.elt,atom.q[0],atom.q[1],atom.q[2]))
            f.close()
        if fmt == 'xsf':
            f=open(prefix+'.xsf','w')
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
            f=open(prefix+'.data','w')
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

    def set_composition(self,composition):
        self.get_element_distribution()
        for elt in composition:
            if elt not in self.pos_elt:
                self.pos_elt[elt]=[]
                logger.info(f"### {elt} {self.pos_elt[elt]} -> stoechiometry {len(self.pos_elt[elt])/len(self.atoms)}")

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





        
    def transform(self,radius=1.0,O=None):
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
    def update_p(self,idx_new,dt=1.0,quench=False):
        self.Ek=0.0
        for atm in self.atoms:
            atm.p=atm.p+0.5*dt*(atm.F[(idx_new+1)%2]+atm.F[idx_new])
            if quench==True:
                if np.dot(atm.F[idx_new],atm.p) <= 0:
                    atm.p=np.zeros(3)
            #self.Ek=self.Ek+CONV*0.5*np.dot(atm.p,atm.p)/atm.mass
            self.Ek=self.Ek+CONV*0.5*np.linalg.norm(atm.p)**2/atm.mass
        self.T=2*self.Ek/(3*len(self.atoms)*KB)
    def update_distances(self):
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
    def update_forces():
        pass

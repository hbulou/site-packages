import numpy as np
import HBPy.Molecule.ForceField

# Atomic masses are based on:
#
#   Meija, J., Coplen, T., Berglund, M., et al. (2016). Atomic weights of
#   the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry,
#   88(3), pp. 265-291. Retrieved 30 Nov. 2016,
#   from doi:10.1515/pac-2015-0305
#
# Standard atomic weights are taken from Table 1: "Standard atomic weights
# 2013", with the uncertainties ignored.
# For hydrogen, helium, boron, carbon, nitrogen, oxygen, magnesium, silicon,
# sulfur, chlorine, bromine and thallium, where the weights are given as a
# range the "conventional" weights are taken from Table 3 and the ranges are
# given in the comments.
# The mass of the most stable isotope (in Table 4) is used for elements
# where there the element has no stable isotopes (to avoid NaNs): Tc, Pm,
# Po, At, Rn, Fr, Ra, Ac, everything after Np
atomic_masses_iupac2016 = np.array([
    1.0,  # X
    1.008,  # H [1.00784, 1.00811]
    4.002602,  # He
    6.94,  # Li [6.938, 6.997]
    9.0121831,  # Be
    10.81,  # B [10.806, 10.821]
    12.011,  # C [12.0096, 12.0116]
    14.007,  # N [14.00643, 14.00728]
    15.999,  # O [15.99903, 15.99977]
    18.998403163,  # F
    20.1797,  # Ne
    22.98976928,  # Na
    24.305,  # Mg [24.304, 24.307]
    26.9815385,  # Al
    28.085,  # Si [28.084, 28.086]
    30.973761998,  # P
    32.06,  # S [32.059, 32.076]
    35.45,  # Cl [35.446, 35.457]
    39.948,  # Ar
    39.0983,  # K
    40.078,  # Ca
    44.955908,  # Sc
    47.867,  # Ti
    50.9415,  # V
    51.9961,  # Cr
    54.938044,  # Mn
    55.845,  # Fe
    58.933194,  # Co
    58.6934,  # Ni
    63.546,  # Cu
    65.38,  # Zn
    69.723,  # Ga
    72.630,  # Ge
    74.921595,  # As
    78.971,  # Se
    79.904,  # Br [79.901, 79.907]
    83.798,  # Kr
    85.4678,  # Rb
    87.62,  # Sr
    88.90584,  # Y
    91.224,  # Zr
    92.90637,  # Nb
    95.95,  # Mo
    97.90721,  # 98Tc
    101.07,  # Ru
    102.90550,  # Rh
    106.42,  # Pd
    107.8682,  # Ag
    112.414,  # Cd
    114.818,  # In
    118.710,  # Sn
    121.760,  # Sb
    127.60,  # Te
    126.90447,  # I
    131.293,  # Xe
    132.90545196,  # Cs
    137.327,  # Ba
    138.90547,  # La
    140.116,  # Ce
    140.90766,  # Pr
    144.242,  # Nd
    144.91276,  # 145Pm
    150.36,  # Sm
    151.964,  # Eu
    157.25,  # Gd
    158.92535,  # Tb
    162.500,  # Dy
    164.93033,  # Ho
    167.259,  # Er
    168.93422,  # Tm
    173.054,  # Yb
    174.9668,  # Lu
    178.49,  # Hf
    180.94788,  # Ta
    183.84,  # W
    186.207,  # Re
    190.23,  # Os
    192.217,  # Ir
    195.084,  # Pt
    196.966569,  # Au
    200.592,  # Hg
    204.38,  # Tl [204.382, 204.385]
    207.2,  # Pb
    208.98040,  # Bi
    208.98243,  # 209Po
    209.98715,  # 210At
    222.01758,  # 222Rn
    223.01974,  # 223Fr
    226.02541,  # 226Ra
    227.02775,  # 227Ac
    232.0377,  # Th
    231.03588,  # Pa
    238.02891,  # U
    237.04817,  # 237Np
    244.06421,  # 244Pu
    243.06138,  # 243Am
    247.07035,  # 247Cm
    247.07031,  # 247Bk
    251.07959,  # 251Cf
    252.0830,  # 252Es
    257.09511,  # 257Fm
    258.09843,  # 258Md
    259.1010,  # 259No
    262.110,  # 262Lr
    267.122,  # 267Rf
    268.126,  # 268Db
    271.134,  # 271Sg
    270.133,  # 270Bh
    269.1338,  # 269Hs
    278.156,  # 278Mt
    281.165,  # 281Ds
    281.166,  # 281Rg
    285.177,  # 285Cn
    286.182,  # 286Nh
    289.190,  # 289Fl
    289.194,  # 289Mc
    293.204,  # 293Lv
    293.208,  # 293Ts
    294.214,  # 294Og
])
atomic_mass=atomic_masses_iupac2016
Z_from_elt={
    "X":0,
    "H":1, "He":2,
    "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    "K":19, "Ca":20, "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27,
    "Ni":28, "Cu":29, "Zn":30,
    "Ga":31, "Ge":32, "As":33, "Se":34, "Br":35, "Kr":36,
    "Rb":37, "Sr":38, "Y":39, "Zr":40, "Nb":41, "Mo":42, "Tc":43, "Ru":44, "Rh":45,
    "Pd":46, "Ag":47, "Cd":48,
    "In":49, "Sn":50, "Sb":51, "Te":52, "I":53, "Xe":54,
    "Cs":55, "Ba":56, "La":57, "Ce":58, "Pr":59, "Nd":60, "Pm":61, "Sm":62, "Eu":63,
    "Gd":64, "Tb":65, "Dy":66,
    "Ho":67, "Er":68, "Tm":69, "Yb":70, "Lu":71,
    "Hf":72, "Ta":73, "W":74, "Re":75, "Os":76, "Ir":77, "Pt":78, "Au":79, "Hg":80,
    "Tl":81, "Pb":82, "Bi":83,
    "Po":84, "At":85, "Rn":86,
    "Fr":87, "Ra":88, "Ac":89, "Th":90, "Pa":91, "U":92, "Np":93, "Pu":84, "Am":85,
    "Cm":96, "Bk":97,
    "Cf":98, "Es":99, "Fm":100, "Md":101, "No":102, "Lr":103,
    "Rf":104, "Db":105, "Sg":106, "Bh":107, "Hs":108, "Mt":109, "Ds":110, "Rg":110,
    "Cn":111, "Nh":112, "Fl":113, "Mc":114,
    "Lv":115, "Ts":116, "Og":117}

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

# Covalent radii from:
#
#  Covalent radii revisited,
#  Beatriz Cordero, Verónica Gómez, Ana E. Platero-Prats, Marc Revés,
#  Jorge Echeverría, Eduard Cremades, Flavia Barragán and Santiago Alvarez,
#  Dalton Trans., 2008, 2832-2838 DOI:10.1039/B801115J
missing=0.2
COV_RADIUS = {
    'H':0.31,  # H
    'He':0.28,  # He
    'Li':1.28,  # Li
    'Be':0.96,  # Be
    'B':0.84,  # B
    'C':0.76,  # C
    'N':0.71,  # N
    'O':0.66,  # O
    'F':0.57,  # F
    'Ne':0.58,  # Ne
    'Na':1.66,  # Na
    'Mg':1.41,  # Mg
    'Al':1.21,  # Al
    'Si':1.11,  # Si
    'P':1.07,  # P
    'S':1.05,  # S
    'Cl':1.02,  # Cl
    'Ar':1.06,  # Ar
    'K':2.03,  # K
    'Ca':1.76,  # Ca
    'Sc':1.70,  # Sc
    'Ti':1.60,  # Ti
    'V':1.53,  # V
    'Cr':1.39,  # Cr
    'Mn':1.39,  # Mn
    'Fe':1.32,  # Fe
    'Co':1.26,  # Co
    'Ni':1.24,  # Ni
    'Cu':1.32,  # Cu
    'Zn':1.22,  # Zn
    'Ga':1.22,  # Ga
    'Ge':1.20,  # Ge
    'As':1.19,  # As
    'Se':1.20,  # Se
    'Br':1.20,  # Br
    'Kr':1.16,  # Kr
    'Rb':2.20,  # Rb
    'Sr':1.95,  # Sr
    'Y':1.90,  # Y
    'Zr':1.75,  # Zr
    'Nb':1.64,  # Nb
    'Mo':1.54,  # Mo
    'Tc':1.47,  # Tc
    'Ru':1.46,  # Ru
    'Rh':1.42,  # Rh
    'Pd':1.39,  # Pd
    'Ag':1.45,  # Ag
    'Cd':1.44,  # Cd
    'In':1.42,  # In
    'Sn':1.39,  # Sn
    'Sb':1.39,  # Sb
    'Te':1.38,  # Te
    'I':1.39,  # I
    'Xe':1.40,  # Xe
    'Cs':2.44,  # Cs
    'Ba':2.15,  # Ba
    'La':2.07,  # La
    'Ce':2.04,  # Ce
    'Pr':2.03,  # Pr
    'Nd':2.01,  # Nd
    'Pm':1.99,  # Pm
    'Sm':1.98,  # Sm
    'Eu':1.98,  # Eu
    'Gd':1.96,  # Gd
    'Tb':1.94,  # Tb
    'Dy':1.92,  # Dy
    'Ho':1.92,  # Ho
    'Er':1.89,  # Er
    'Tm':1.90,  # Tm
    'Yb':1.87,  # Yb
    'Lu':1.87,  # Lu
    'Hf':1.75,  # Hf
    'Ta':1.70,  # Ta
    'W':1.62,  # W
    'Re':1.51,  # Re
    'Os':1.44,  # Os
    'Ir':1.41,  # Ir
    'Pt':1.36,  # Pt
    'Au':1.36,  # Au
    'Hg':1.32,  # Hg
    'Tl':1.45,  # Tl
    'Pb':1.46,  # Pb
    'Bi':1.48,  # Bi
    'Po':1.40,  # Po
    'At':1.50,  # At
    'Rn':1.50,  # Rn
    'Fr':2.60,  # Fr
    'Ra':2.21,  # Ra
    'Ac':2.15,  # Ac
    'Th':2.06,  # Th
    'Pa':2.00,  # Pa
    'U':1.96,  # U
    'Np':1.90,  # Np
    'Pu':1.87,  # Pu
    'Am':1.80,  # Am
    'Cm':1.69,  # Cm
    'Bk':missing,  # Bk
    'Cf':missing,  # Cf
    'Es':missing,  # Es
    'Fm':missing,  # Fm
    'Md':missing,  # Md
    'No':missing,  # No
    'Lr':missing,  # Lr
    'Rf':missing,  # Rf
    'Db':missing,  # Db
    'Sg':missing,  # Sg
    'Bh':missing,  # Bh
    'Hs':missing,  # Hs
    'Mt':missing,  # Mt
    'Ds':missing,  # Ds
    'Rg':missing,  # Rg
    'Cn':missing,  # Cn
    'Nh':missing,  # Nh
    'Fl':missing,  # Fl
    'Mc':missing,  # Mc
    'Lv':missing,  # Lv
    'Ts':missing,  # Ts
    'Og':missing  # Og
}

# Couleurs/rayons très simples pour tes éléments
# CPK_COLOR = {
#     'H': (1.0, 1.0, 1.0),
#     'C': (0.2, 0.2, 0.2),
#     'N': (0.2, 0.2, 1.0),
#     'O': (1.0, 0.0, 0.0),
#     'Pt': (0.8, 0.8, 0.9),
#     'Pd': (0.8, 0.8, 0.8),
#     'Rh': (0.7, 0.8, 0.9),
#     'Ir': (0.8, 0.8, 0.7),
#     'Ru': (0.7, 0.7, 0.9),
# }

# couleurs RGBA par élément
CPK_COLOR = {
    # Éléments organiques courants
    "H":  (1.0, 1.0, 1.0),        # blanc
    "C":  (0.35, 0.35, 0.35),     # gris CPK (plus lisible que noir pur)
    "N":  (0.19, 0.31, 0.97),     # bleu CPK
    "O":  (1.0, 0.05, 0.05),      # rouge CPK
    "F":  (0.56, 0.88, 0.31),     # vert clair CPK
    "P":  (1.0, 0.5, 0.0),        # orange
    "S":  (1.0, 1.0, 0.19),       # jaune CPK
    "Cl": (0.12, 0.94, 0.12),     # vert CPK

    # Halogènes lourds
    "Br": (0.65, 0.16, 0.16),     # brun rouge
    "I":  (0.58, 0.0, 0.58),      # violet foncé

    # Métaux — version réaliste proche CPK / Jmol
    "Fe": (224/255, 140/255, 61/255),     # fer – orange foncé
    "Co": (200/255, 0/255, 200/255),     # fer – orange foncé
    "Ni": (255/255, 255/255, 0/255),     # fer – orange foncé
    "Cu": (212/255, 123/255, 57/255),     # cuivre – brun cuivré
    "Zn": (0.49, 0.50, 0.69),     # zinc – gris bleu
    "Au": (255/255, 214/255, 0.0),       # or
    "Ag": (191/255,191/255,191/255),     # argent
    "Pt": (100/255, 100/255, 100/255),     # platine – gris acier

    # Métaux de transition du groupe du platine 
    "Ru": (255/255, 0/255, 0/255),     # ruthénium
    "Rh": (6/255, 118/255, 132/255),     # rhodium
    "Pd": (100/255, 100/255,0/255),     # palladium
    "Os": (201/255, 176/255, 156/255),     # osmium
    "Ir": (0/255, 0/255, 255/255),     # iridium

}    




class Atom:
    def __init__(self,elt='H',q=np.zeros(3),p=np.zeros(3),idx=0,pbc=[-1,-1,-1],
                 mass=1.0,
                 save=True):
        self.elt=elt
        self.q=q
        self.p=p
        self.mass=atomic_mass[chemical_symbols.index(elt)]
        self.idx=idx
        self.pbc=pbc
        self.R=[]
        self.d=[]
        self.F=np.array([np.zeros(3),np.zeros(3)])
        self.idx_neigh=[]
        self.Erep=0.0
        self.Eattsqr=0.0
        self.Etot=0.0
        self.Eb=0.0

    def distance_from_(self,q=np.zeros(3)):
        d=0.0
        for i in range(3):
            d+=(self.q[i]-q[i])**2
        return np.sqrt(d)
    def duplicate(self):
        """Crée une copie indépendante de manière optimisée."""
        # 1. On crée une nouvelle instance avec les paramètres de base copiés
        new_atom = Atom(
            elt=self.elt,
            q=np.copy(self.q),
            p=np.copy(self.p),
            idx=self.idx,
            pbc=self.pbc.copy() if self.pbc is not None else None,
            mass=self.mass,
            save=True # ou self.save si vous avez gardé cet attribut
        )
        
        # 2. On copie les attributs internes (états calculés)
        new_atom.R = self.R.copy()
        new_atom.d = self.d.copy()
        new_atom.F = np.copy(self.F)
        new_atom.idx_neigh = self.idx_neigh.copy()
        new_atom.Erep = self.Erep
        new_atom.Eattsqr = self.Eattsqr
        new_atom.Etot = self.Etot
        new_atom.Eb = self.Eb
        
        return new_atom

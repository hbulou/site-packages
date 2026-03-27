import numpy as np

class Atom:
    def __init__(self,q=np.zeros(3),p=np.zeros(3),idx=0,pbc=[-1,-1,-1]):
        self.q=q
        self.p=p
        self.idx=idx
        self.pbc=pbc
        self.R=[]
        self.d=[]
        self.F=np.array([np.zeros(3),np.zeros(3)])
        self.idx_neigh=[]

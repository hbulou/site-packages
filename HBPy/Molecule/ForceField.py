import numpy as np

class TBSMA():
    def __init__(self,A=0.0,p=0.0,xi=0.0,q=0.0,r0=0.0):
        self.A=A
        self.p=p
        self.xi=xi
        self.q=q
        self.r0=r0
    def __repr__(self):
        return f"{self.p:.3f}"

class ForceField():
    def __init__(self,list_elt=['Pt','Pd','Rh','Ir','Ru']):
        self.tbsma = {}
        mixte={}
        mixte['Pd','Pd']=np.array([0.25692558,9.26323299,1.96029713,4.31347750,2.80638616])
        mixte['Pt','Pt']=np.array([0.31097458,9.71202562,2.76019839,3.80067087,2.81574078])
        mixte['Cu','Cu']=np.array([0.16281169,8.55808897,1.61981305,2.86919407,2.56316603])
        mixte['Rh','Rh']=np.array([0.37089308,8.76574095,3.09242925,3.63233602,2.70909811])
        mixte['Ru','Ru']=np.array([0.46564728,8.24125464,4.06953316,3.23441902,2.69413071])
        mixte['Ag','Ag']=np.array([0.09282080,10.97475169,1.07487532,3.24354562,2.93735085])
        mixte['Ir','Ir']=np.array([0.57012997,8.56367003,4.47310405,3.81288174,2.75025844])
        for i in range(len(list_elt)):
            elti=list_elt[i]
            for j in range(len(list_elt)):
                eltj=list_elt[j]
                if (elti, eltj) not in mixte:
                    mixte[elti,eltj]=np.zeros(5)
                    mixte[eltj,elti]=np.zeros(5)
                    for k in range(5):
                        mixte[elti,eltj][k]=0.5*(mixte[elti,elti][k]+mixte[eltj,eltj][k])
                        mixte[eltj,elti][k]=mixte[elti,eltj][k]

                self.tbsma[elti,eltj]=TBSMA(A =mixte[elti,eltj][0],
                                            p =mixte[elti,eltj][1],
                                            xi=mixte[elti,eltj][2],
                                            q =mixte[elti,eltj][3],
                                            r0=mixte[elti,eltj][4])            
                    



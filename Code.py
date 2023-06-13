import numpy as np

def Uncertenty(l,m,d,f):
    Ue = np.sqrt(((0.002**2)*(-4*1.6067*(l**3)*m*(f**2)/(d**5))**2) +((1.6067*(l**3)*(f**2)/(d**4))**2)*(0.01**2)+((3*1.6067*(l**2)*(f**2)*m/(d**4))**2)*(0.01**2)+((2*1.6067*(l**3)*f*m/(d**4))**2))
    print('Ue = ',Ue)
    return Ue

def E(l,m,d,f,t):
    E = t*1.6067*(l**3)*m*(f**2)/(d**4)
    print('E = ',E)
    return E

def divide(a,b):
    print('a/b = ',a/b)
    return a/b

divide(Uncertenty(199.7,29.81,4.852,568.5),E(199.7,29.81,4.852,568.5,1.002))
divide(Uncertenty(199.92,32.76,4.833,376.25),E(199.92,32.76,4.833,376.25,1.0032))
divide(Uncertenty(200.22,10.75,4.868,570.5),E(200.22,10.75,4.868,570.5,1.0032))
divide(Uncertenty(199.84,15.21,5.828,671.75),E(199.84,15.21,5.828,671.75,1.0047))
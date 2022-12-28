import numpy as np
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import pdb

# TODO: make this part a function to determine all constants as a fxn of gam, mdot, rs

gam= 5./3
mdot=1.
rs=np.power(10.,2.5) #1000. # 8. #

n= 1./(gam-1.)
uc= np.sqrt(mdot/(2.*rs))
Vc= -np.sqrt(np.power(uc,2.)/(1.-3.*np.power(uc,2)))
Tc= -n*np.power(Vc,2)/((n+1)*(n*np.power(Vc,2)-1.))
C1= uc*np.power(rs,2)*np.power(Tc,n)
C2=np.power(1.+(1.+n)*Tc,2.)*(1.-2.*mdot/rs+np.power(C1,2)/(np.power(rs,4)*np.power(Tc,2*n)))
uprime=C1/(np.power(rs,2)*np.power(Tc,n))
C2prime=(-2.*mdot/rs+np.power(uprime,2.))+(2.*(1.+n)*Tc+np.power((1.+n)*Tc,2))*(1.-2.*mdot/rs+np.power(uprime,2))

def get_Tfunc(T,r,C1,C2,n):
    #result = np.power(1.+(1.+n)*T,2.)*(1.-2.*mdot/r+np.power(C1/(np.power(r,2.)*np.power(T,n)),2))-C2
    utemp=C1/(np.power(r,2.)*np.power(T,n))
    result = (-2*mdot/r+np.power(utemp,2))+(2.*(1.+n)*T+np.power((1.+n)*T,2))*(1.-2.*mdot/r+np.power(utemp,2))-C2prime
    return result

def get_T(r, C1, C2, n, ax=None, inflow_sol=True):
    rtol = 1.e-12
    ftol = 1.e-14 #5e-15 #
    Tinf = (np.sqrt(C2)-1.)/(n+1)
    Tapprox=np.power(C1*np.sqrt(2./np.power(r,3)),1./n)
    
    bounds1=[Tinf, Tapprox] # smaller T solution
    bounds2=[np.fmax(Tapprox, Tinf),1.] # larger T solution
    if inflow_sol:
        if (r<rs):
            Tmin=bounds1[0]
            Tmax=bounds1[1]
        else:
            Tmin=bounds2[0]
            Tmax=bounds2[1]
    else:
        if (r<rs):
            Tmin=bounds1[1]
            Tmax=1.5
        else:
            Tmin=1e-10 #0.1*Tinf
            Tmax=bounds2[0]
      

    #print(C2, C2prime)
    #print(Tapprox, Tinf)
    #if Tapprox>5*Tinf:
    #    print("choosing previous bound")
    #else:
    #    print("choosing Tinf")

    
    if ax is not None:
        print("plotting")
        T_arr=np.linspace(Tmin,Tmax,100)
        ax.plot(T_arr,get_Tfunc(T_arr,r,C1,C2,n),'k:')
        ax.axhline(0)
        ax.set_yscale('symlog')

    T0=Tmin
    f0=get_Tfunc(T0,r,C1,C2,n)
    T1=Tmax
    f1=get_Tfunc(T1,r,C1,C2,n)

    if (f0*f1>0):
        print("error")
        return -1

    Th = 0.5*(T0+T1)
    fh = get_Tfunc(Th,r,C1,C2,n)

    epsT= rtol*(Tmin+Tmax)


    while ((abs(Th-T0)>epsT) and (abs(Th-T1)>epsT)) and (abs(fh)>ftol):
        if (fh*f0>0): # bisection method # < a more efficient way
            T0=Th
            f0=fh
        else:
            T1=Th
            f1=fh

        Th=(T1-T0)/2.+T0 #(f1*T0-f0*T1)/(f1-f0)
        fh= get_Tfunc(Th,r,C1,C2,n)
    
    #print(r, Tmin, Tmax, Th)
    return Th

def get_quantity_for_rarr(rarr,quantity):
    Tarr=np.array([get_T(r,C1,C2,n) for r in rarr])
    rhoarr=np.power(Tarr,n)
    if quantity=='T':
        return Tarr
    elif quantity=='rho':
        return rhoarr
    elif quantity=='ur' or quantity=='u^r': #
        urarr=C1/(np.power(rarr,2)*np.power(Tarr,n))
        return urarr
    elif quantity=='vr':
    #elif quantity=='u^r':
        urarr=C1/(np.power(rarr,2)*np.power(Tarr,n))
        vrarr=urarr/np.sqrt(1.-2.*mdot/rarr+np.power(urarr,2))
        return vrarr
    elif quantity=='u':
        uarr=rhoarr*Tarr*n
        return uarr
    else:
        return None

def _main():
    rarr= np.logspace(np.log10(2.1),np.log10(1e9),100)

    if 0:
        Tarr=[]
        for r in rarr:
            Tarr += [get_T(r,C1,C2,n,inflow_sol=True)]

        Tarr= np.array(Tarr)
        rhoarr=np.power(Tarr,n)

        #plt.loglog(rarr,rhoarr)
        plt.loglog(rarr,Tarr)
        plt.savefig("./temp.png")
    else:
        fig,ax=plt.subplots(1,1,figsize=(8,6))
        #print(uc,Vc,Tc,C1,C2,C2prime)
        T=get_T(1e8,C1,C2,n,ax,inflow_sol=True)
        rho=np.power(T,n)
        u=rho*T*n
        print("T = {:.5g}, rho = {:.5g}, u = {:.5g}".format(T,rho,u))
        print("logrho = {:.6g}, loguoverrho = {:.6g}".format(np.log10(rho),np.log10(u/rho)))
        plt.savefig("./temp.png")

if __name__ == "__main__":
    _main()

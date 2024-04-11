# Wersja 1.0 z dnia 12.03.2024

import numpy as np
from scipy.stats import norm

def cc_imp_curves(FXspot,tenor,yf,rate_ccy,swapp):
    rt     = yf*rate_ccy
    df_ccy = 1/(1+rt)
    FXfwd  = FXspot+swapp/10000
    df_pln = FXspot/FXfwd*df_ccy
    rate_pln = (1/df_pln-1)/yf
    curves=[
         tenor,
         yf,
         df_ccy,
         rate_ccy,
         df_pln,
         rate_pln
    ]
    return curves
    
def cc_imp_curves2(FXspot,tenor,days,rate_ccy,ccy_basis, pln_basis,swapp):
    yf_ccy=days/ccy_basis
    yf_pln=days/pln_basis
    rt     = yf_ccy*rate_ccy
    df_ccy = 1/(1+rt)
    FXfwd  = FXspot+swapp/10000
    df_pln = FXspot/FXfwd*df_ccy
    rate_pln = (1/df_pln-1)/yf_pln
    curves=[
         tenor,
         yf_ccy,
         df_ccy,
         rate_ccy,
         yf_pln,
         df_pln,
         rate_pln
    ]
    return curves
    
    
def int_df(t,yf,df):
    i_max = len(yf)-1
    if t<=yf[0]:
        dft = df[0]**(t/yf[0])
    elif t>yf[i_max]:
        dft = df[i_max]**(t/yf[i_max])
    else:
        i = 0
        yfi = yf[i]
        while t > yfi:
            i+=1
            yfi = yf[i]    
        tau = (t-yf[i-1])/(yf[i]-yf[i-1])
        dft = df[i-1]**(1-tau)*df[i]**tau
    return dft    

##-----------------------------------------------

def BinaryPay(omega, S, K):
    if omega*(S-K)>=0:
        val = 1.0
    else:
        val = 0.0
    return val

def VanillaPay(omega, S, K):
    return max(omega*(S-K), 0)

##-------------------------------------------------

def BS_value(fx_spot,df_ccy_t,df_pln_t,tau,sigma,omega,strike,nominal):   
    fx_fwd = fx_spot*df_ccy_t/df_pln_t
    d1 = (np.log(fx_fwd/strike)+0.5*sigma**2*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    value = nominal*omega*(df_ccy_t*fx_spot*norm.cdf(omega*d1)-df_pln_t*strike*norm.cdf(omega*d2))
    return value

def BS_Binary(omega, FXspot, sigma, df_pln, df_ccy, strike, tau, N,curr):
    d1 = (np.log(FXspot*df_ccy/(strike*df_pln))+0.5*sigma**2*tau)/(np.sqrt(tau)*sigma)
    d2 = d1 - sigma*np.sqrt(tau)
    if curr=="nonbase": 
        value=N*df_pln*norm.cdf(omega*d2)
    elif curr=="base":
        value=N*FXspot*df_ccy*norm.cdf(omega*d1)
    return value


# Greki - współczynniki wrażliwości opcji waniliowych z formuł BS

def DeltaBS(omega,FXspot, sigma, df_pln, df_ccy, strike, tau, N):
    d1 = (np.log(FXspot*df_ccy/(strike*df_pln))+0.5*sigma**2*tau)/(np.sqrt(tau)*sigma)
    return N*omega*df_ccy*norm.cdf(omega*d1)

def GammaBS(FXspot, sigma, df_pln, df_ccy, strike, tau, N):
    d1 = (np.log(FXspot*df_ccy/(strike*df_pln))+0.5*sigma**2*tau)/(np.sqrt(tau)*sigma)
    return N*df_ccy*norm.pdf(d1)/(FXspot*np.sqrt(tau)*sigma)

def VegaBS(FXspot, sigma, df_pln, df_ccy, strike, tau, N):
    d1 = (np.log((FXspot*df_ccy)/(strike*df_pln))+0.5*sigma**2*tau)/(np.sqrt(tau)*sigma)
    return N*FXspot*df_ccy*np.sqrt(tau)*norm.pdf(d1)
    
def VolgaBS(FXspot, sigma, df_pln, df_ccy, strike, tau, N):
    d1 = (np.log((FXspot*df_ccy)/(strike*df_pln))+0.5*sigma**2*tau)/(np.sqrt(tau)*sigma)
    d2 = d1 - sigma*np.sqrt(tau)
    return N*FXspot*df_ccy*np.sqrt(tau)*norm.pdf(d1)*d1*d2/sigma

def VannaBS(FXspot, sigma, df_pln, df_ccy, strike, tau, N):
    d1 = (np.log((FXspot*df_ccy)/(strike*df_pln))+0.5*sigma**2*tau)/(np.sqrt(tau)*sigma)
    d2 = d1 - sigma*np.sqrt(tau)
    return -N*df_ccy*norm.pdf(d1)*d2/sigma
    
# ----------------------------------------------------------------
    
def Delta_pi_BS(spot_fwd, omega, FXspot, sigma, df_pln, df_ccy, strike, tau, N):
    FXfwd=FXspot*df_ccy/df_pln
    d2 = (np.log(FXfwd/strike)-0.5*sigma**2*tau)/(np.sqrt(tau)*sigma)
    if spot_fwd=="spot":
    	delta=N*omega*strike*df_ccy*norm.cdf(omega*d2)/FXfwd
    elif spot_fwd=="fwd":
    	delta=N*omega*strike*norm.cdf(omega*d2)/FXfwd
    return delta
    
def StrikeFromDelta(spot_fwd,prem_incl,omega,delta,FXspot,sigma,df_pln,df_ccy,T):
    eps=0.00001
    Imax=10
    i=0
    K1=FXspot*df_ccy/df_pln
    K=0
    dK=0.001*K1
    if prem_incl=="pi": # pi - delta z premią
        while abs(K1-K)>eps and i<Imax:
            K=K1
            delta1=Delta_pi_BS(spot_fwd,omega, FXspot, sigma, df_pln, df_ccy, K1, T, 1)
            delta_p=Delta_pi_BS(spot_fwd,omega, FXspot, sigma, df_pln, df_ccy, K1+dK, T, 1)
            delta_m=Delta_pi_BS(spot_fwd,omega, FXspot, sigma, df_pln, df_ccy, K1-dK, T, 1)
            d_delta=(delta_p-delta_m)/(2*dK)
            K1-=(delta1-delta)/d_delta
            i+=1
    elif prem_incl=="std": # std - delta bez premii
        F=FXspot*df_ccy/df_pln
        if spot_fwd=="spot": df=df_ccy
        elif spot_fwd=="fwd": df=1
        K=F*np.exp(-omega*norm.ppf(omega*delta/df)*sigma*np.sqrt(T)+0.5*sigma**2*T)  
    return K
    
# Implikowanie zmienności z ceny opcji  

def ImpVol(Value_mkt,omega,FXspot,df_d,df_f,K,T): # sigma0 - punkt startowy 
    FXfwd=FXspot*df_f/df_d
    if Value_mkt<max(0,omega*(FXspot*df_f-K*df_d)) or (omega==1 and 
    Value_mkt>FXspot*df_f) or (omega==-1 and Value_mkt>K*df_d): sigma=0
    else:
        saddle_point=np.sqrt(2*abs(np.log(FXfwd/K))/T)
        if saddle_point > 0: sigma = 0.9*saddle_point
        else: sigma = 0.1
        Nmax=100
        eps1=0.0001
        eps2=1e-06
        sigma1=sigma+1.1*eps1
        Value_BS=Value_mkt+1.1*eps2
        i=1
        while abs(sigma1-sigma)>eps1 and abs(Value_mkt-Value_BS)>eps2 and i<Nmax:
            sigma1=sigma
            Value_BS=BS_value(omega, FXspot, sigma, df_d, df_f, K, T, 1)
            Vega_BS=VegaBS(FXspot, sigma, df_d, df_f, K, T, 1)
            sigma-=(Value_BS-Value_mkt)/Vega_BS
            i+=1
    return sigma

# ----------------------------------------------------------------

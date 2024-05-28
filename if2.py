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

def BS_value(fx_spot,df_pln_t,df_ccy_t,tau,sigma,omega,strike,nominal): 
    fx_fwd = fx_spot*df_ccy_t/df_pln_t
    d1 = (np.log(fx_fwd/strike)+0.5*sigma**2*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    value = nominal*omega*(df_ccy_t*fx_spot*norm.cdf(omega*d1)-df_pln_t*strike*norm.cdf(omega*d2))
    return value

def ValueBS(omega,FXspot, sigma, df_pln, df_ccy, strike, tau, N): 
    FX_fwd = FXspot*df_ccy/df_pln
    d1 = (np.log(FX_fwd/strike)+0.5*sigma**2*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    value = N*omega*(df_ccy*FXspot*norm.cdf(omega*d1)-df_pln*strike*norm.cdf(omega*d2))
    return value

def BS_Binary(omega, FXspot, sigma, df_pln, df_ccy, strike, tau, N,curr):
    d1 = (np.log(FXspot*df_ccy/(strike*df_pln))+0.5*sigma**2*tau)/(np.sqrt(tau)*sigma)
    d2 = d1 - sigma*np.sqrt(tau)
    if curr=="nonbase": 
        value=N*df_pln*norm.cdf(omega*d2)
    elif curr=="base":
        value=N*FXspot*df_ccy*norm.cdf(omega*d1)
    return value

def BS_OT(eta, paytime, FXspot, sigma, df_pln, df_ccy, barrier, tau, R):
    r_d=-np.log(df_pln)/tau
    r_f=-np.log(df_ccy)/tau
    df=(1-paytime)+paytime*df_pln
    if eta*(FXspot-barrier)<=0: return df*R
 #   theta_p=(r_d-r_f)/sigma+0.5*sigma
    theta_m=(r_d-r_f)/sigma-0.5*sigma
    theta=np.sqrt(theta_m**2+2*(1-paytime)*r_d)
    d_p = (np.log(FXspot/barrier)-sigma*theta*tau)/(np.sqrt(tau)*sigma)
    d_m = (-np.log(FXspot/barrier)-sigma*theta*tau)/(np.sqrt(tau)*sigma)
    BdivS=barrier/FXspot
    value=R*df*(BdivS**((theta_m+theta)/sigma)*norm.cdf(-eta*d_p)+BdivS**((theta_m-theta)/sigma)*norm.cdf(eta*d_m))
    return value

def Touch_PDF(t, FXspot, sigma, r_d, r_f, barrier):
    theta_m = (r_d-r_f)/sigma-0.5*sigma
    x0 = np.log(barrier/FXspot)/sigma
    prob_dens = x0/(t*np.sqrt(2*np.pi*t))*np.exp(-(x0-theta_m*t)**2/(2.0*t))
    return prob_dens

def Touch_CDF(t, FXspot, sigma, r_d, r_f, barrier):
    theta_m = (r_d-r_f)/sigma-0.5*sigma
    x0 = np.log(barrier/FXspot)/sigma
    dp = (theta_m*t-x0)/np.sqrt(t)
    dm = (-theta_m*t-x0)/np.sqrt(t)
    t_cdf = norm.cdf(dp)+np.exp(2.0*x0*theta_m*t)*norm.cdf(dm)
    return t_cdf

def Exp_touch_moment(FXspot, sigma, r_d, r_f, barrier):
    theta_m = (r_d-r_f)/sigma-0.5*sigma
    x0 = np.log(barrier/FXspot)/sigma
    if theta_m >=0:
        exp_tm = x0/theta_m
    else:
        exp_tm = x0*np.exp(2.0*x0*theta_m)/(-theta_m)
    return exp_tm

# Opcje barierowe

def BSF(omega,eta,X,Y,sig,r_d,r_f,d,T):
    df_d=np.exp(-r_d*T)
    df_f=np.exp(-r_f*T)
    return omega*(df_f*X*norm.cdf(eta*d)-df_d*Y*norm.cdf(eta*(d-np.sqrt(T)*sig)))
def d(X,Y,sig,r_d,r_f,T):
    return (np.log(X/Y)+(r_d-r_f+0.5*sig**2)*T)/(sig*np.sqrt(T))
 
def BS1(omega,S,K,sig,r_d,r_f,T):
    d1 = d(S,K,sig,r_d,r_f,T)
    return BSF(omega,omega,S,K,sig,r_d,r_f,d1,T)

def BS2(omega,S,K,B,sig,r_d,r_f,T):
    d2 = d(S,B,sig,r_d,r_f,T)
    return BSF(omega,omega,S,K,sig,r_d,r_f,d2,T)

def BS3(omega,eta,S,K,B,sig,r_d,r_f,T):
    d3 = d(B**2/S,K,sig,r_d,r_f,T)
    nu = 2*(r_d-r_f)/(sig**2)
    R  = (B/S)**(nu-1)
    return R*BSF(omega,eta,B**2/S,K,sig,r_d,r_f,d3,T)

def BS4(omega,eta,S,K,B,sig,r_d,r_f,T):
    d4 = d(B,S,sig,r_d,r_f,T)
    nu = 2*(r_d-r_f)/(sig**2)
    R  = (B/S)**(nu-1)
    return R*BSF(omega,eta,B**2/S,K,sig,r_d,r_f,d4,T)

def Value_UIC(FXspot,sig,r_d,r_f,Strike,Barrier,T):
    if FXspot<Barrier:            # bariera nie przekroczona, opcja nie aktywowana
        if Strike>Barrier:
            value=BS1(1,FXspot,Strike,sig,r_d,r_f,T)
        else:
            V2=BS2(1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            V3=BS3(1,-1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            V4=BS4(1,-1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            value=V2-V3+V4
    else:                         # bariera przekroczona, opcja aktywowana
        value=BS1(1,FXspot,Strike,sig,r_d,r_f,T)
    return value

def Value_UIP(FXspot,sig,r_d,r_f,Strike,Barrier,T):
    if FXspot<Barrier:            # bariera nie przekroczona, opcja nie aktywowana
        if Strike>Barrier:
            V1=BS1(-1,FXspot,Strike,sig,r_d,r_f,T)
            V2=BS2(-1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            V4=BS4(-1,-1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            value=V1-V2+V4
        else:
            value=BS3(-1,-1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
    else:                         # bariera przekroczona, opcja aktywowana
        value=BS1(-1,FXspot,Strike,sig,r_d,r_f,T)
    return value

def Value_DIP(FXspot,sig,r_d,r_f,Strike,Barrier,T):
    if FXspot>Barrier:            # bariera nie przekroczona, opcja nie aktywowana
        if Strike>Barrier:
            V2=BS2(-1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            V3=BS3(-1,1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            V4=BS4(-1,1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            value=V2-V3+V4
        else:
            value=BS1(-1,FXspot,Strike,sig,r_d,r_f,T)
    else:                          # bariera przekroczona, opcja aktywowana
        value=BS1(-1,FXspot,Strike,sig,r_d,r_f,T)
    return value

def Value_DOP(FXspot,sig,r_d,r_f,Strike,Barrier,T):
    if FXspot > Barrier:            # bariera nie przekroczona, opcja aktywna
        if Strike > Barrier:
            V1 = BS1(-1,FXspot,Strike,sig,r_d,r_f,T)
            V2 = BS2(-1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            V3 = BS3(-1,1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            V4 = BS4(-1,1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            value = V1-V2+V3-V4
        else:
            value=BS1(-1,FXspot,Strike,sig,r_d,r_f,T)
    else:                          # bariera przekroczona, opcja nie aktywna
        value=0.0
    return value


def Value_DIC(FXspot,sig,r_d,r_f,Strike,Barrier,T):
    if FXspot>Barrier:             # bariera nie przekroczona, opcja nie aktywowana
        if Strike>Barrier:
            value = BS3(1,1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
        else:
            V1 = BS1(1,FXspot,Strike,sig,r_d,r_f,T)
            V2 = BS2(1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            V4 = BS4(1,1,FXspot,Strike,Barrier,sig,r_d,r_f,T)
            value = V1-V2+V4
    else:                          # bariera przekroczona, opcja aktywowana
        value=BS1(1,FXspot,Strike,sig,r_d,r_f,T)
    return value

def Value_DOC(FXspot,sig,r_d,r_f,Strike,Barrier,T):
    Vannila = BS1(1,FXspot,Strike,sig,r_d,r_f,T)
    DIC     = Value_DIC(FXspot,sig,r_d,r_f,Strike,Barrier,T)
    return Vannila-DIC

def Value_UOC(FXspot,sig,r_d,r_f,Strike,Barrier,T):
    Vannila=BS1(1,FXspot,Strike,sig,r_d,r_f,T)
    UIC=Value_UIC(FXspot,sig,r_d,r_f,Strike,Barrier,T)
    return Vannila-UIC

def Value_DOP1(FXspot,sig,r_d,r_f,Strike,Barrier,T):
    Vannila = BS1(-1,FXspot,Strike,sig,r_d,r_f,T)
    DIP     = Value_DIP(FXspot,sig,r_d,r_f,Strike,Barrier,T)
    return Vannila-DIP

def Value_UOP(FXspot,sig,r_d,r_f,Strike,Barrier,T):
    Vannila=BS1(-1,FXspot,Strike,sig,r_d,r_f,T)
    UIP=Value_UIP(FXspot,sig,r_d,r_f,Strike,Barrier,T)
    return Vannila-UIP

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
    
def Delta_pi_BS(is_spot_delta, omega, FXspot, sigma, df_pln, df_ccy, strike, tau, N):
    FXfwd = FXspot*df_ccy/df_pln
    d2    = (np.log(FXfwd/strike)-0.5*sigma**2*tau)/(np.sqrt(tau)*sigma)
    df = df_ccy if is_spot_delta else 1.0	
    delta = N*omega*strike*df*norm.cdf(omega*d2)/FXfwd
    return delta

def DeltaBS_quoted(is_spot,is_pi,omega,fx_spot,sigma,df_d,df_f,tau,strike,nominal):   
    fx_fwd = fx_spot*df_f/df_d
    d1 = (np.log(fx_fwd/strike)+0.5*sigma**2*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau) 
    df = df_f if is_spot else 1.0
    if is_pi:
        delta = nominal*omega*(strike/fx_fwd)*df*norm.cdf(omega*d2)
    else:    
        delta = nominal*omega*df*norm.cdf(omega*d1)
    return delta
    
def StrikeFromDelta(is_spot_delta,is_pi,omega,delta,FXspot,sigma,df_pln,df_ccy,tau):
    if omega*delta <= 0.0:
        print("złe dane")
        return "NaN"
    eps=0.00001
    Imax=10
    i=0
    K1=FXspot*df_ccy/df_pln
    K=0
    dK=0.001*K1
    if is_pi: # is_pi = True - delta z premią, False - delta bez premii
        while abs(K1-K)>eps and i<Imax:
            K = K1
            delta1  = Delta_pi_BS(is_spot_delta, omega, FXspot, sigma, df_pln, df_ccy, K1, tau, 1)
            delta_p = Delta_pi_BS(is_spot_delta, omega, FXspot, sigma, df_pln, df_ccy, K1+dK, tau, 1)
            delta_m = Delta_pi_BS(is_spot_delta, omega, FXspot, sigma, df_pln, df_ccy, K1-dK, tau, 1)
            d_delta = (delta_p-delta_m)/(2*dK)
            K1 -= (delta1-delta)/d_delta
            i += 1
    else:
        F = FXspot*df_ccy/df_pln 
        df = df_ccy if is_spot_delta else 1.0
        K = F*np.exp(-omega*norm.ppf(omega*delta/df)*sigma*np.sqrt(tau)+0.5*sigma**2*tau)  
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
            Value_BS=ValueBS(omega, FXspot, sigma, df_d, df_f, K, T, 1)
            Vega_BS=VegaBS(FXspot, sigma, df_d, df_f, K, T, 1)
            sigma-=(Value_BS-Value_mkt)/Vega_BS
            i+=1
    return sigma

def ATM_Strike(type_atm,is_pi,FXspot,sigma_atm,df_d,df_f,tau):
    FXfwd = FXspot*df_f/df_d
    if type_atm.upper() == "ATM-S":
        K_ATM = FXspot
    elif type_atm.upper() == "ATM-F":
        K_ATM = FXfwd
    elif type_atm.upper() == "ATM-0D":
        eta   = 1-2*float(is_pi)                       # is_pi = True - delta z premią 
        K_ATM = FXfwd*np.exp(eta*0.5*sigma_atm**2*tau)
    return K_ATM

# ----------------------------------------------------------------
# Interpolacja Vanna-Volga - I rzędu 

# I rzędu 
def vv_int1s(K,smile_data):
    K1,sig1,K2,sig2,K3,sig3 = smile_data
    x1 = (np.log(K/K2)*np.log(K/K3))/(np.log(K1/K2)*np.log(K1/K3))
    x2 = (np.log(K/K1)*np.log(K/K3))/(np.log(K2/K1)*np.log(K2/K3))
    x3 = (np.log(K/K1)*np.log(K/K2))/(np.log(K3/K1)*np.log(K3/K2))
    sigmaK = x1*sig1 + x2*sig2 + x3*sig3
    return sigmaK

# II rzędu
# Uwaga: przy założeniu, że sig_BS=sig_2, zwykle sig_atm
def vv_int2s(K,smile_data,FXfwd,T): 
    K1,sig1,K2,sig2,K3,sig3 = smile_data
    x1 = (np.log(K/K2)*np.log(K/K3))/(np.log(K1/K2)*np.log(K1/K3))
    x3 = (np.log(K/K1)*np.log(K/K2))/(np.log(K3/K1)*np.log(K3/K2))
    sig1K = vv_int1s(K,smile_data)
    d1K   = (np.log(FXfwd/K)+0.5*sig2**2*T)/(sig2*np.sqrt(T))
    d2K   = d1K - sig2*np.sqrt(T)
    d12K  = d1K*d2K
    d1K1  = (np.log(FXfwd/K1) + 0.5*sig2**2*T)/(sig2*np.sqrt(T))
    d2K1  = d1K1 - sig2*np.sqrt(T)
    d12K1 = d1K1*d2K1
    d1K3  = (np.log(FXfwd/K3)+0.5*sig2**2*T)/(sig2*np.sqrt(T))
    d2K3  = d1K3 - sig2*np.sqrt(T)
    d12K3 = d1K3*d2K3
    nu1   = sig1K - sig2
    nu2   = x1*d12K1*(sig1 - sig2)**2 + x3*d12K3*(sig3 - sig2)**2
    nu    = sig2**2 + d12K*(2*sig2*nu1 + nu2)
    sigma2K = sig2 + (-sig2 + np.sqrt(nu))/d12K
    return sigma2K

# II rzędu
# Uwaga: bez założenia, że sig_BS=sig_2, zwykle sig_atm 
def vv_int22s(K,K1,smile_data,sig_BS,FXfwd,T):
    K1,sig1,K2,sig2,K3,sig3 = smile_data
    x1=(np.log(K/K2)*np.log(K/K3))/(np.log(K1/K2)*np.log(K1/K3))
    x2=(np.log(K/K1)*np.log(K/K3))/(np.log(K2/K1)*np.log(K2/K3))  
    x3=(np.log(K/K1)*np.log(K/K2))/(np.log(K3/K1)*np.log(K3/K2))
    sig1K=vv_int1s(K,K1,sig1,K2,sig2,K3,sig3)
    d1K=(np.log(FXfwd/K)+0.5*sig_BS**2*T)/(sig_BS*np.sqrt(T))
    d2K=d1K-sig_BS*np.sqrt(T)
    d12K=d1K*d2K
    d1K1=(np.log(FXfwd/K1)+0.5*sig_BS**2*T)/(sig_BS*np.sqrt(T))
    d2K1=d1K1-sig_BS*np.sqrt(T)
    d12K1=d1K1*d2K1
    d1K2=(np.log(FXfwd/K2)+0.5*sig_BS**2*T)/(sig_BS*np.sqrt(T))
    d2K2=d1K2-sig_BS*np.sqrt(T)
    d12K2=d1K2*d2K2
    d1K3=(np.log(FXfwd/K3)+0.5*sig_BS**2*T)/(sig_BS*np.sqrt(T))
    d2K3=d1K3-sig_BS*np.sqrt(T)
    d12K3=d1K3*d2K3
    nu1=sig1K-sig_BS
    nu2=x1*d12K1*(sig1-sig_BS)**2+x2*d12K2*(sig2-sig_BS)**2+x3*d12K3*(sig3-sig_BS)**2
    nu=sig_BS**2+d12K*(2*sig_BS*nu1+nu2)
    sigma2K=sig_BS+(-sig_BS+np.sqrt(nu))/d12K
    return sigma2K

# Wycena Vanna-Volga

# Over Hedge jest względem wyceny przy zmienności sig_atm
def VV_value(omega,FXspot,df_d,df_f,K,T,N,smile_data):
    K_25P,sig_25p,K_ATM,sig_atm,K_25C,sig_25c = smile_data
    P1_mkt=ValueBS(-1, FXspot, sig_25p, df_d, df_f, K_25P, T, 1)
    P1_BS =ValueBS(-1, FXspot, sig_atm, df_d, df_f, K_25P, T, 1)
    OH1=P1_mkt-P1_BS
    C3_mkt= ValueBS(1, FXspot, sig_25c, df_d, df_f, K_25C, T, 1)
    C3_BS = ValueBS(1, FXspot, sig_atm, df_d, df_f, K_25C, T, 1)
    OH3=C3_mkt-C3_BS
    VegaK1 = VegaBS(FXspot, sig_atm, df_d, df_f, K_25P, T, 1)
    VegaK3 = VegaBS(FXspot, sig_atm, df_d, df_f, K_25C, T, 1)
    VegaK  = VegaBS(FXspot, sig_atm, df_d, df_f, K, T, 1)
    x1=(np.log(K/K_ATM)*np.log(K/K_25C))/(np.log(K_25P/K_ATM)*np.log(K_25P/K_25C))*(VegaK/VegaK1)
    x3=(np.log(K/K_25P)*np.log(K/K_ATM))/(np.log(K_25C/K_25P)*np.log(K_25C/K_ATM))*(VegaK/VegaK3)
    Value_BS=ValueBS(omega, FXspot, sig_atm, df_d, df_f, K, T, 1)
    value=N*(Value_BS+x1*OH1+x3*OH3)
    return value

# Metoda ogólniejsza niż VV_value. Over Hedge jest względem wyceny przy zmienności sig_BS a nie sig_atm
def VV_value2(omega,FXspot,df_d,df_f,K,T,N,smile_data,sig_BS):
    K_25P,sig_25p,K_ATM,sig_atm,K_25C,sig_25c = smile_data
    P1_mkt=ValueBS(-1, FXspot, sig_25p, df_d, df_f, K_25P, T, 1)
    P1_BS =ValueBS(-1, FXspot, sig_BS,  df_d, df_f, K_25P, T, 1)
    OH1=P1_mkt-P1_BS
    C2_mkt= ValueBS(1, FXspot, sig_atm, df_d, df_f, K_ATM, T, 1)
    C2_BS = ValueBS(1, FXspot, sig_BS,  df_d, df_f, K_ATM, T, 1)
    OH2=C2_mkt-C2_BS
    C3_mkt= ValueBS(1, FXspot, sig_25c, df_d, df_f, K_25C, T, 1)
    C3_BS = ValueBS(1, FXspot, sig_BS,  df_d, df_f, K_25C, T, 1)
    OH3=C3_mkt-C3_BS
    VegaK1 = VegaBS(FXspot, sig_BS, df_d, df_f, K_25P, T, 1)
    VegaK2 = VegaBS(FXspot, sig_BS, df_d, df_f, K_ATM, T, 1)
    VegaK3 = VegaBS(FXspot, sig_BS, df_d, df_f, K_25C, T, 1)
    VegaK  = VegaBS(FXspot, sig_BS, df_d, df_f, K, T, 1)
    x1=(np.log(K/K_ATM)*np.log(K/K_25C))/(np.log(K_25P/K_ATM)*np.log(K_25P/K_25C))*(VegaK/VegaK1)
    x2=(np.log(K/K_25P)*np.log(K/K_25C))/(np.log(K_ATM/K_25P)*np.log(K_ATM/K_25C))*(VegaK/VegaK2)
    x3=(np.log(K/K_25P)*np.log(K/K_ATM))/(np.log(K_25C/K_25P)*np.log(K_25C/K_ATM))*(VegaK/VegaK3)
    Value_BS=ValueBS(omega, FXspot, sig_BS, df_d, df_f, K, T, 1)
    value=N*(Value_BS+x1*OH1+x2*OH2+x3*OH3)
    return value

# Obliczanie kosztów ryzyka zmienności
def VolRiskCost(FXspot,df_d,df_f,T,smile_data):
    K_25P,sig_25p,K_ATM,sig_atm,K_25C,sig_25c = smile_data
    P1_mkt = ValueBS(-1, FXspot, sig_25p, df_d, df_f, K_25P, T, 1)
    P1_BS  = ValueBS(-1, FXspot, sig_atm, df_d, df_f, K_25P, T, 1)
    OH1 = P1_mkt - P1_BS
    OH2 = 0
    C3_mkt = ValueBS(1, FXspot, sig_25c, df_d, df_f, K_25C, T, 1)
    C3_BS  = ValueBS(1, FXspot, sig_atm, df_d, df_f, K_25C, T, 1)
    OH3 = C3_mkt - C3_BS
    VegaK1  = VegaBS(FXspot, sig_atm, df_d, df_f, K_25P, T, 1)
    VegaK2  = VegaBS(FXspot, sig_atm, df_d, df_f, K_ATM, T, 1)
    VegaK3  = VegaBS(FXspot, sig_atm, df_d, df_f, K_25C, T, 1)
    VannaK1 = VannaBS(FXspot, sig_atm, df_d, df_f, K_25P, T, 1)
    VannaK2 = VannaBS(FXspot, sig_atm, df_d, df_f, K_ATM, T, 1)
    VannaK3 = VannaBS(FXspot, sig_atm, df_d, df_f, K_25C, T, 1)
    VolgaK1 = VolgaBS(FXspot, sig_atm, df_d, df_f, K_25P, T, 1)
    VolgaK2 = VolgaBS(FXspot, sig_atm, df_d, df_f, K_ATM, T, 1)
    VolgaK3 = VolgaBS(FXspot, sig_atm, df_d, df_f, K_25C, T, 1)

    VV=np.array([[VegaK1,VannaK1,VolgaK1],
                [VegaK2,VannaK2,VolgaK2],
                [VegaK3,VannaK3,VolgaK3]])
    
    OH=np.array([OH1,OH2,OH3])
    vrc=np.linalg.inv(VV).dot(OH)
    return vrc

# Wycena via koszty ryzyka zmienności
def VV_value3(omega,FXspot,df_d,df_f,K,T,sig_atm,Vol_risk_cost):
    VegaK     = VegaBS(FXspot, sig_atm, df_d, df_f, K, T, 1)
    VannaK    = VannaBS(FXspot, sig_atm, df_d, df_f, K, T, 1)
    VolgaK    = VolgaBS(FXspot, sig_atm, df_d, df_f, K, T, 1)
    Vol_sensi = np.array([VegaK,VannaK,VolgaK])
    OH        = Vol_risk_cost.dot(Vol_sensi)
    Value_BS  = ValueBS(omega, FXspot, sig_atm, df_d, df_f, K, T, 1)
    value     = Value_BS+OH
    return value
    
# ----------------------------------------------------------------

# Interpolacja Vanna-Volga - I rzędu / poprzednie wersje z długą listą argumentów

# I rzędu 
def vv_int1(K,K1,sig1,K2,sig2,K3,sig3):
    x1=(np.log(K/K2)*np.log(K/K3))/(np.log(K1/K2)*np.log(K1/K3))
    x2=(np.log(K/K1)*np.log(K/K3))/(np.log(K2/K1)*np.log(K2/K3))
    x3=(np.log(K/K1)*np.log(K/K2))/(np.log(K3/K1)*np.log(K3/K2))
    sigmaK=x1*sig1+x2*sig2+x3*sig3
    return sigmaK

# II rzędu
# Uwaga: przy założeniu, że sig_BS=sig_2, zwykle sig_atm
def vv_int2(K,K1,sig1,K2,sig2,K3,sig3,FXfwd,T): 
    x1=(np.log(K/K2)*np.log(K/K3))/(np.log(K1/K2)*np.log(K1/K3))
    x3=(np.log(K/K1)*np.log(K/K2))/(np.log(K3/K1)*np.log(K3/K2))
    sig1K=vv_int1(K,K1,sig1,K2,sig2,K3,sig3)
    d1K=(np.log(FXfwd/K)+0.5*sig2**2*T)/(sig2*np.sqrt(T))
    d2K=d1K-sig2*np.sqrt(T)
    d12K=d1K*d2K
    d1K1=(np.log(FXfwd/K1)+0.5*sig2**2*T)/(sig2*np.sqrt(T))
    d2K1=d1K1-sig2*np.sqrt(T)
    d12K1=d1K1*d2K1
    d1K3=(np.log(FXfwd/K3)+0.5*sig2**2*T)/(sig2*np.sqrt(T))
    d2K3=d1K3-sig2*np.sqrt(T)
    d12K3=d1K3*d2K3
    nu1=sig1K-sig2
    nu2=x1*d12K1*(sig1-sig2)**2+x3*d12K3*(sig3-sig2)**2
    nu=sig2**2+d12K*(2*sig2*nu1+nu2)
    sigma2K=sig2+(-sig2+np.sqrt(nu))/d12K
    return sigma2K

# II rzędu
# Uwaga: bez założenia, że sig_BS=sig_2, zwykle sig_atm 
def vv_int22(K,K1,sig1,K2,sig2,K3,sig3,sig_BS,FXfwd,T): 
    x1=(np.log(K/K2)*np.log(K/K3))/(np.log(K1/K2)*np.log(K1/K3))
    x2=(np.log(K/K1)*np.log(K/K3))/(np.log(K2/K1)*np.log(K2/K3))  
    x3=(np.log(K/K1)*np.log(K/K2))/(np.log(K3/K1)*np.log(K3/K2))
    sig1K=vv_int1(K,K1,sig1,K2,sig2,K3,sig3)
    d1K=(np.log(FXfwd/K)+0.5*sig_BS**2*T)/(sig_BS*np.sqrt(T))
    d2K=d1K-sig_BS*np.sqrt(T)
    d12K=d1K*d2K
    d1K1=(np.log(FXfwd/K1)+0.5*sig_BS**2*T)/(sig_BS*np.sqrt(T))
    d2K1=d1K1-sig_BS*np.sqrt(T)
    d12K1=d1K1*d2K1
    d1K2=(np.log(FXfwd/K2)+0.5*sig_BS**2*T)/(sig_BS*np.sqrt(T))
    d2K2=d1K2-sig_BS*np.sqrt(T)
    d12K2=d1K2*d2K2
    d1K3=(np.log(FXfwd/K3)+0.5*sig_BS**2*T)/(sig_BS*np.sqrt(T))
    d2K3=d1K3-sig_BS*np.sqrt(T)
    d12K3=d1K3*d2K3
    nu1=sig1K-sig_BS
    nu2=x1*d12K1*(sig1-sig_BS)**2+x2*d12K2*(sig2-sig_BS)**2+x3*d12K3*(sig3-sig_BS)**2
    nu=sig_BS**2+d12K*(2*sig_BS*nu1+nu2)
    sigma2K=sig_BS+(-sig_BS+np.sqrt(nu))/d12K
    return sigma2K

# Funkcja celu na potrzeby kalibracji interpolacji
def S_value(K_25p_stgl,K_25c_stgl,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,tau,sig_atm,sig_rr,sig):

    sig_25p = sig_atm - 0.5*sig_rr + sig
    sig_25c = sig_atm + 0.5*sig_rr + sig

    K_ATM = ATM_Strike(type_atm,is_pi,FXrate,sig_atm,df_d,df_f,tau)
    K_25p = StrikeFromDelta(is_spot_delta,is_pi,-1,-0.25,FXrate,sig_25p,df_d,df_f,tau)
    K_25c = StrikeFromDelta(is_spot_delta,is_pi, 1, 0.25,FXrate,sig_25c,df_d,df_f,tau)
   
    FXfwd=FXrate*df_f/df_d

    smile_data   = (K_25p,sig_25p,K_ATM,sig_atm,K_25c,sig_25c)
    sig_25p_stgl = vv_int1s(K_25p_stgl,smile_data)
    sig_25c_stgl = vv_int1s(K_25c_stgl,smile_data)
    #sig_stgl_25p=vv_int2(K_25p_stgl,smile_data,FXfwd,tau)
    #sig_stgl_25c=vv_int2(K_25c_stgl,smile_data,FXfwd,tau)
    V_25p=ValueBS(-1, FXrate, sig_25p_stgl, df_d, df_f, K_25p_stgl, tau, 1)
    V_25c=ValueBS( 1, FXrate, sig_25c_stgl, df_d, df_f, K_25c_stgl, tau, 1)
    value=V_25c+V_25p-STGL_mkt_val
    return value

# Pochodna funcji celu
def dS_value(K_stgl_25p,K_stgl_25c,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,T,sig_atm,sig_rr,sig):
    dsig=0.0001
    S0=S_value(K_stgl_25p,K_stgl_25c,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,T,sig_atm,sig_rr,sig)
    S1p=S_value(K_stgl_25p,K_stgl_25c,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,T,sig_atm,sig_rr,sig+dsig)
    S1m=S_value(K_stgl_25p,K_stgl_25c,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,T,sig_atm,sig_rr,sig-dsig)
    dS=(S1p-S1m)/(2*dsig)
    return dS

# Kalibracja intrepolacji
def SmileCaliber(K_stgl_25p,K_stgl_25c,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,T,sig_atm,sig_rr,sig_bf):
    eps1=0.00001
    eps2=0.00001
    Nmax=100
    sig=sig_bf
    sig1=1.1*sig
    n=1
    S=S_value(K_stgl_25p,K_stgl_25c,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,T,sig_atm,sig_rr,sig)
    Sprime=dS_value(K_stgl_25p,K_stgl_25c,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,T,sig_atm,sig_rr,sig)
    while abs(sig1-sig)>eps1 and abs(S)>eps2 and n<Nmax:
        sig1=sig
        sig-=S/Sprime
        S=S_value(K_stgl_25p,K_stgl_25c,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,T,sig_atm,sig_rr,sig)
        Sprime=dS_value(K_stgl_25p,K_stgl_25c,STGL_mkt_val,type_atm,is_spot_delta,is_pi,FXrate,df_d,df_f,T,sig_atm,sig_rr,sig)
        n+=1
#    print(n)
#    print(S)
    return sig
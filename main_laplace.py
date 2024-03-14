import pandas as pd
import numpy as np
from math import sqrt
from scipy.optimize import minimize
from scipy.stats import laplace, chi2
import matplotlib.pyplot as plt 
from tqdm import tqdm

from main_statDes import getReturn

def main():
    #On importe les données
    # fileBtc = 'BTCUSD.xls'
    # fileLtc = 'LTCUSD.xls'
    # fileEth = 'ETHUSD.xls'

    # dfBtc = pd.read_excel(fileBtc)
    # dfLtc = pd.read_excel(fileLtc)
    # dfEth = pd.read_excel(fileEth)
    filedfCac = 'CAC40.xls'
    filedfNas = 'NASDAQ.xls'

    dfCac = pd.read_excel(filedfCac)
    dfNas = pd.read_excel(filedfNas)

    dfCac['Returns'] = dfCac['Close'].pct_change().dropna() * 100
    dfNas['Returns'] = dfNas['Close'].pct_change().dropna() * 100

    #On calcule les rendements pour chaque cryptoactif    
    # dfBtc, dfLtc, dfEth = getReturn(dfBtc, dfLtc, dfEth)

    '''Ci-dessous, calcul des VaR des trois modèles sur les rendements, modifier la VaR, les intervalles et le dataframe pour chaque VaR
        Intervalle de returns à prendre pour chaque cryptomonnaie qui correspond au testing set pour chaque actif:
            -> BTC : [997:3119]
            -> LTC : [999:1924]
            -> ETH : [499:1272]
        Modifier également le dataframe dans la fonction prev_var pour chaque modèle
    '''
    
    returns = dfCac['Returns']
    alpha = 0.05
    length = 200

    #L-GAS:

    optimisation = optimize_fixed_df(returns)
    coef,  starting_scale = optimisation.x
    laplace_gas  = compute_gas(returns, coef,starting_scale)

    _,predicted_var = prev_var(dfCac, optimize_fixed_df, training_len=length,quantile=alpha)
 
    sum = np.sum(identify_var_violations(returns[200:1560], predicted_var))
    print("Pourcentage de violations:",sum/len(returns[200:1560]))
    lr_ucd_laplace, p_val_ucd_laplace = lr_ucd(sum, len(returns[200:1560]), alpha)
    print("Statistique de test de Kupiec:",lr_ucd_laplace)
    print("P-val du test de Kupiec:",p_val_ucd_laplace)

    compare_var(returns[997:3119], predicted_var)

    #L-GAS(p)

    # optimisation = optimize_fixed_dfp(returns)
    # coef, p, starting_scale = optimisation.x
    # laplacep_gas = compute_gasp(returns, coef, p, starting_scale)

    # gas,predicted_var = prev_varp(dfCac, optimize_fixed_dfp, training_len=length, quantile=alpha)

    # sum = np.sum(identify_var_violations(returns[200:1560], predicted_var))
    # print("Pourcentage de violations:",sum/len(returns[200:1560]))
    # lr_ucd_laplacep, p_val_ucd_laplacep = lr_ucd(sum, len(returns[200:1560]), alpha)
    # print("Statistique de test de Kupiec:",lr_ucd_laplacep)
    # print("P-val du test de Kupiec:",p_val_ucd_laplacep)

    # #L-GAS(pt)

    # optimisation = optimize_fixed_dfpt(returns)
    # coef_1, coef_2, coef_3, p, starting_scale = optimisation.x
    # laplacept_gas = compute_gaspt(returns, coef_1, coef_2, coef_3,p,starting_scale)

    # gas,predicted_var = prev_varpt(dfCac, optimize_fixed_dfpt,training_len=length, quantile=alpha)

    # sum = np.sum(identify_var_violations(returns[200:1560], predicted_var))
    # print("Pourcentage de violations:",sum/len(returns[200:1560]))
    # lr_ucd_laplacept, p_val_ucd_laplacept = lr_ucd(sum, len(returns[200:1560]), alpha)
    # print("Statistique de test de Kupiec:",lr_ucd_laplacept)
    # print("P-val du test de Kupiec:",p_val_ucd_laplacept)

    return 0

'''Fonction pour calculer la prochaine valeur du paramètre scale pour une loi de Laplace'''
def get_next_scale(ret: float, prev_scale: float, coef:float):

    prev_scale = sqrt(np.abs(prev_scale))
    new_scale = coef*prev_scale**2 + (1-coef)*sqrt(2)*np.abs(ret)*sqrt(prev_scale)

    return new_scale

'''Fonction pour calculer la densité pour une loi de Laplace'''
def get_pdf(ret: float, scale: float):
    try:
        density = 1/(sqrt(2)*sqrt(scale)) * np.exp(-(sqrt(2)*np.abs(ret)) / sqrt(scale))
    except ValueError:
        density=1e-10
    except ZeroDivisionError:
        density=1e-10
    return density

'''Fonction pour calculer notre modèle GAS pour une loi de Laplace'''
def compute_gas(returns: pd.Series, coef:float, starting_scale:float) -> pd.DataFrame:

    gas = {"pdf": [], "scale": []}
    returns.dropna(inplace=True)

    #Paramètres initiaux
    gas["scale"].append(starting_scale*1e-1)
    pdf = get_pdf(returns.iloc[0], starting_scale)
    gas["pdf"].append(pdf)

    #Calcul du modèle
    returns_list = returns.to_list()
    for i, ret in enumerate(returns_list[1:]):  # We start at 1 because we already initialized the model.
        new_scale = get_next_scale(ret=ret,prev_scale=gas["scale"][i],coef=coef)
        new_pdf = get_pdf(ret, new_scale) if get_pdf(ret, new_scale) < 50  else 1e-2

        gas["scale"].append(new_scale*1e-1)
        gas["pdf"].append(new_pdf)
    
    return pd.DataFrame(gas)

'''Fonction à optimizer pour avoir le meilleur scale pour une loi de Laplace'''
def optimize_fixed_df(returns: pd.Series):

    def get_log_likelihood(params: list[float]) -> float:
        """
        params: parameters to optimize 
        Returns the log-likelyhood * -1
        """
        coef, starting_scale = params
        gas = compute_gas(returns, coef, starting_scale)
        log_pdf = np.log(gas["pdf"])
        log_pdf = np.nan_to_num(log_pdf, nan=1e-9)  
        return -np.sum(log_pdf)
    

    optimization = minimize( 
        fun=get_log_likelihood,
        x0=[0.1, 0.4],
        bounds=((0.0, 1.0), (0.01, 0.99)),
    )
    return optimization

'''Fonction pour calculer la VaR pour une loi de Laplace'''
def get_var(scale:pd.Series, returns:pd.Series, alpha:float, date:pd.Series):
    returns.dropna(inplace=True)
    var = laplace.ppf(alpha, scale = scale*1e1)
    for i in range(1,len(var)):
        if var[i]<-100:var[i]=var[i-1]
    var = pd.Series(var, index=date[:len(var)])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(date,returns, 'b-', label='Returns', linewidth=0, markersize=1, marker='o')
    ax1.plot(date,var, 'r-', label='Variance', linewidth=1)
    ax1.set_xlabel('Date')
    plt.title('Returns and Variance Evolution L-GAS')
    plt.show()

    return var

'''Fonction pour faire nos prédictions de la VaR pour une loi de Laplace'''
def prev_var(returns: pd.DataFrame, optimize_fixed_df: callable, training_len: int = 500, quantile: float = 0.01):
    previsions = []
    previsions_index = []
    for i in tqdm(range(training_len, len(returns))):
        optimisation = optimize_fixed_df(returns["Returns"].iloc[:i])
        coef, scale = optimisation.x
        trained = compute_gas(returns["Returns"].iloc[:i], coef, scale)
        last_values = trained.iloc[-1]
        anticipated_scale = get_next_scale(returns['Returns'].iloc[i],last_values["scale"],last_values["pdf"])
        anticipated_var = laplace.ppf(quantile, scale=anticipated_scale*1e1)
        previsions.append(anticipated_var)
        previsions_index.append(returns.index[i-1])
    for i in range(1,len(previsions)):
        if previsions[i]<-100:previsions[i]=previsions[i-1]

    return trained, pd.Series(previsions, index=previsions_index)

'''Fonction pour grapher la VaR de notre modèle et la VaR simulée pour une loi de Laplace'''
def compare(gas:pd.DataFrame, var:pd.Series):

    gas['scale'].loc[gas['scale'] < -100] = -100
    var.loc[var < -100] = -100

    plt.figure(figsize=(10,6))
    ax = plt.gca()
    _var = laplace.ppf(0.05, scale=gas['scale']*1e1)
    pd.Series(_var, index=gas.index).plot(ax=ax, title="Value at Risk", color="red")
    var.plot(ax=ax)
    ax.legend()
    plt.show()

'''Fonction pour grapher la VaR de notre modèle avec les rendements'''
def compare_var(returns:pd.Series,predicted_var:pd.Series):

    returns = returns.where(returns >= -100, other=returns.shift(2))
    predicted_var = predicted_var.where(predicted_var >= -100, other=predicted_var.shift(2))

    plt.plot(returns[15:], label='Returns')
    plt.plot(predicted_var[15:], label='Predicted Var')
    plt.legend()
    plt.show()

    plt.show()
    
    
'''Fonction pour faire le test LR_ucd'''
def lr_ucd(n:int, t:int, alpha:float):

    first_term = np.log((n/t)**n * (1-n/t)**(t-n))
    if first_term == 'inf':first_term=1
    second_term = np.log((1-alpha)**(t-n) * alpha**n)
    if second_term == 'inf':second_term=1
    
    lr_ucd = 2*(first_term - second_term)
    p_val_ucd = chi2.sf(lr_ucd,1)

    return lr_ucd, p_val_ucd

'''Identifie les violations de la VaR.'''
def identify_var_violations(returns, var):

    returns = returns.reindex(var.index)
    violations = returns < var

    return violations.astype(int).tolist()

'''Fonction pour calculer le prochain scalepour une loi de Laplace(p)'''
def get_next_scalep(ret: float, p:float, prev_scale: float, coef:float):

    k = sqrt(p**2 + (1-p)**2)
    term_k = k/(1-p) if ret >= 0 else k/p
    prev_scale = sqrt(np.abs(prev_scale))

    new_scale = coef*prev_scale**2 + (1-coef)*prev_scale*np.abs(ret)*term_k

    return new_scale

'''Fonction pour calculer la densité pour une loi de Laplace(p)'''
def get_pdfp(ret: float, p:float, scale: float):
    try:
        term_k= 1/(1-p) if ret >= 0 else 1/p
        k = sqrt(p**2 + (1-p)**2)
        density = k/scale * np.exp(-term_k * (k*np.abs(ret)/scale))
    except ValueError:
        density=1e-10
    except ZeroDivisionError:
        density=1e-10
    return density

'''Fonction pour calculer notre modèle GAS pour une loi de Laplace(p)'''
def compute_gasp(returns: pd.Series, coef:float, p:float, starting_scale:float) -> pd.DataFrame:

    gas = {"pdf": [], "scale": [], "p":[]}
    returns.dropna(inplace=True)

    # Paramètres intitiaux
    gas["scale"].append(starting_scale*1e-1)
    pdf = get_pdfp(returns.iloc[0], p, starting_scale)
    gas["pdf"].append(pdf)
    gas["p"].append(p)

    # Calcul du modèle
    returns_list = returns.to_list()
    for i, ret in enumerate(returns_list[1:]):  # We start at 1 because we already initialized the model.
        new_scale = get_next_scalep(ret=ret, p=p, prev_scale=gas["scale"][i],coef=coef)
        new_pdf = get_pdfp(ret, p, new_scale) if get_pdfp(ret,p, new_scale) < 50  else 1e-2

        gas["scale"].append(new_scale*1e-1)
        gas["pdf"].append(new_pdf)
        gas["p"].append(p)
    
    return pd.DataFrame(gas)

'''Fonction à optimizer pour avoir le meilleur scale pour une loi de Laplace(p)'''
def optimize_fixed_dfp(returns: pd.Series):

    def get_log_likelihoodp(params: list[float]) -> float:
        """
        params: parameters to optimize 
        Returns the log-likelyhood * -1
        """
        coef, p, starting_scale = params
        gas = compute_gasp(returns, coef,p, starting_scale)
        log_pdf = np.log(gas["pdf"])
        log_pdf = np.nan_to_num(log_pdf, nan=1e-9)  
        return -np.sum(log_pdf)
    

    optimization = minimize( 
        fun=get_log_likelihoodp,
        x0=[0.1, 0.2,  0.4],
        bounds=((0.0, 1.0), (0.001, 0.999), (0.01, 0.99)),
    )
    return optimization

'''Fonction pour calculer la VaR pour une loi de Laplace(p)'''
def get_varp(scale:pd.Series, p:pd.Series, returns:pd.Series, alpha:float, date:pd.Series):
    returns.dropna(inplace=True)
    var = laplace.ppf(q=alpha, scale=scale*1e2)
    for i in range(1,len(var)):
        if var[i]<-100:var[i]=var[i-1]
    var = pd.Series(var, index=date[:len(var)])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(date,returns, 'b-', label='Returns', linewidth=0, markersize=1, marker='o')
    ax1.plot(date,var, 'r-', label='Variance', linewidth=1)
    ax1.set_xlabel('Date')
    plt.title('Returns and Variance Evolution')
    plt.show()

    return var
    
'''Fonction pour faire nosprédictions de la VaR pour une loi de Laplace(p)'''
def prev_varp(returns: pd.DataFrame, optimize_fixed_dfp: callable, training_len: int = 500, quantile: float = 0.01):
    previsions = []
    previsions_index = []
    for i in tqdm(range(training_len, len(returns))):
        optimisation = optimize_fixed_dfp(returns["Returns"].iloc[:i])
        coef, p, scale = optimisation.x
        trained = compute_gasp(returns["Returns"].iloc[:i], coef,p, scale)
        last_values = trained.iloc[-1]
        anticipated_scale = get_next_scalep(returns['Returns'].iloc[i],p,last_values["scale"],coef)
        anticipated_var = laplace.ppf(q=quantile, scale=anticipated_scale*1e1, loc=p)
        previsions.append(anticipated_var)
        previsions_index.append(returns.index[i-1])
    for i in range(1,len(previsions)):
        if previsions[i]<-100:previsions[i]=previsions[i-1]

    return trained, pd.Series(previsions, index=previsions_index)


'''Fonction pour calculer le prochain scale pour une loi de Laplace(pt)'''
def get_next_scalept(ret: float, p:float, prev_scale: float, coef_1:float, coef_2:float, coef_3:float,prev_u:float, prev_v:float):
    u = coef_2*prev_u+(1-coef_2)*np.abs(ret) if ret >=0 else coef_2*prev_u
    v = coef_3*prev_v+(1-coef_3)*np.abs(ret) if ret <0 else coef_3*prev_v
    if v==0:v=prev_v
    p = 1/(1+sqrt(u/v))
    k = sqrt(p**2 + (1-p)**2)
    term_k = k/(1-p) if ret >= 0 else k/p
    prev_scale = sqrt(np.abs(prev_scale))

    new_scale = coef_1*prev_scale**2 + (1-coef_1)*prev_scale*np.abs(ret)*term_k

    return new_scale, p, term_k,u,v

'''Fonction pour calculer la densité pour une loi de Laplace(pt)'''
def get_pdfpt(ret: float, p:float, scale: float,k:float):
    try:
        term_k= 1/(1-p) if ret >= 0 else 1/p
        density = k/scale * np.exp(-term_k * (k*np.abs(ret)/scale))
    except ValueError:
        density=1e-10
    except ZeroDivisionError:
        density=1e-10
    return density

'''Fonction pour calculer notre modèle GAS pour une loi de Laplace(pt)'''
def compute_gaspt(returns: pd.Series, coef_1:float, coef_2:float, coef_3:float, p:float, starting_scale:float) -> pd.DataFrame:

    gas = {"pdf": [], "scale": [], "p":[], "u":[], "v":[],"k":[]}
    returns.dropna(inplace=True)

    #Paramètres initiaux
    gas["scale"].append(starting_scale*1e-1)
    pdf = get_pdfpt(returns.iloc[0], p, starting_scale,k=sqrt(p**2 + (1-p)**2))
    gas["pdf"].append(pdf)
    gas["p"].append(p)
    gas["u"].append(0)
    gas["v"].append(0.01)
    gas["k"].append(p**2 + (1-p)**2)

    #Calcul du modèle
    returns_list = returns.to_list()
    for i, ret in enumerate(returns_list[1:]):  # We start at 1 because we already initialized the model.
        new_scale, p, k,u,v = get_next_scalept(ret=ret, p=gas["p"][i], prev_scale=gas["scale"][i],coef_1=coef_1, coef_2=coef_2, coef_3=coef_3, prev_u=gas["u"][i], prev_v=gas["v"][i])
        new_pdf = get_pdfpt(ret, p, new_scale,k) if get_pdfpt(ret,p, new_scale,k) < 50  else 1e-2

        gas["scale"].append(new_scale*1e-1)
        gas["pdf"].append(new_pdf)
        gas["p"].append(p)
        gas["k"].append(k)
        gas["u"].append(u)
        gas["v"].append(v)
    
    return pd.DataFrame(gas)

'''Fonction à optimizer pour avoir le meilleur scale pour notre loi pour une loi de Laplace(pt)'''
def optimize_fixed_dfpt(returns: pd.Series):

    def get_log_likelihoodpt(params: list[float]) -> float:
        """
        params: parameters to optimize
        Returns the log-likelyhood * -1
        """
        coef_1, coef_2, coef_3, p, starting_scale = params
        gas = compute_gaspt(returns, coef_1, coef_2, coef_3 ,p, starting_scale)
        log_pdf = np.log(gas["pdf"])
        log_pdf = np.nan_to_num(log_pdf, nan=1e-9)  
        return -np.sum(log_pdf)
    
    optimization = minimize( 
        fun=get_log_likelihoodpt,
        x0=[0.1,0.1, 0.1, 0.5,  0.4],
        bounds=((0.0, 1.0),(0.0, 1.0),(0.0, 1.0), (0.001, 0.999), (0.01, 0.99)),
    )
    return optimization

'''Fonction pour calculer la VaR pour une loi de Laplace(pt)'''
def get_varpt(scale:pd.Series, p:pd.Series, returns:pd.Series, alpha:float, date:pd.Series):
    returns.dropna(inplace=True)
    var = laplace.ppf(q=alpha, scale=scale*1e3)
    for i in range(1,len(var)):
        if var[i]<-100:var[i]=var[i-1]
    var = pd.Series(var, index=date[:len(var)])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(date,returns, 'b-', label='Returns', linewidth=0, markersize=1, marker='o')
    ax1.plot(date,var, 'r-', label='Variance', linewidth=1)
    ax1.set_xlabel('Date')
    plt.title('Returns and Variance Evolution')
    plt.show()

    return var

'''Fonction pour faire nosprédictions de la VaR pour une loi de Laplace(pt)'''
def prev_varpt(returns: pd.DataFrame, optimize_fixed_dfpt: callable, training_len: int = 500, quantile: float = 0.01):
    previsions = []
    for i in range(len(returns)):
        if returns["Returns"].iloc[i]==0: returns["Returns"].iloc[i]=1e-1
    previsions_index = []
    for i in tqdm(range(training_len, len(returns))):
        optimisation = optimize_fixed_dfpt(returns["Returns"].iloc[:i])
        coef_1, coef_2, coef_3, p, scale = optimisation.x
        trained = compute_gaspt(returns["Returns"].iloc[:i], coef_1, coef_2, coef_3,p, scale)
        last_values = trained.iloc[-1]
        anticipated_scale,p,_,_,_ = get_next_scalept(returns['Returns'].iloc[i],p,last_values["scale"],coef_1, coef_2, coef_3,last_values["u"],last_values["v"])
        anticipated_var = laplace.ppf(q=quantile,scale=anticipated_scale*1e2, loc=p)
        previsions.append(anticipated_var)
        previsions_index.append(returns.index[i-1])
    for i in range(1,len(previsions)):
        if previsions[i]<-100:previsions[i]=previsions[i-1]

    return trained, pd.Series(previsions, index=previsions_index)

main()
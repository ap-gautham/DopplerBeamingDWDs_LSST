import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.io import fits

import astropy
import nifty_ls
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle
from astropy.coordinates import SkyCoord
from astropy import units

from tqdm import tqdm
import scipy

import multiprocessing
import pandas as pd
import sqlite3
import lightkurve
import lightkurve.periodogram

import h5py
import time
import os
import sys
sys.path.append('/home/gadaman1/data_znadia1/02_Packages/')
import WD_models

HR_grid = (-0.6, 1.25, 0.002, 7, 15, 0.01)
model_DA  = WD_models.load_model(low_mass_model='Bedard2020',
                             middle_mass_model='Bedard2020',
                             high_mass_model='Bedard2020', 
                              atm_type='H', 
                              HR_bands=('bp3-rp3', 'G3'),
                              HR_grid=HR_grid,)

mass_logteff_logg_DA = WD_models.interp_xy_z_func(x=model_DA['mass_array'],\
                                               y=model_DA['logteff'],\
                                               z=model_DA['logg'],\
                                               interp_type='linear')


###################

dbfile = '/home/gadaman1/data_znadia1/01_Projects/01_WD/07_LSST_DWD/baseline_v5.0.0_10yrs.fits'
df_table = Table(astropy.io.fits.open(dbfile)[1].data)
df_table['field_coordinates'] = astropy.coordinates.SkyCoord(df_table['fieldRA']*units.deg,df_table['fieldDec']*units.deg)

###################

band_wavelengths = {
    "u": 3570*units.Angstrom,   # nm
    "g": 4767*units.Angstrom,   # nm
    "r": 6215*units.Angstrom,   # nm
    "i": 7545*units.Angstrom,   # nm
    "z": 8701*units.Angstrom,  # nm
    "y": 10004*units.Angstrom  # nm
}

band_error = \
{
    'u': {'gamma': 0.038, 'm5': 23.78},
    'g': {'gamma': 0.039, 'm5': 24.81},
    'r': {'gamma': 0.039, 'm5': 24.35},
    'i': {'gamma': 0.039, 'm5': 23.92},
    'z': {'gamma': 0.039, 'm5': 23.34},
    'y': {'gamma': 0.039, 'm5': 22.45}
}

bands = ['u','g','r','i','z','y']

def sigma1(m,band,m5=None):
    sigma_sys = 0.005

    if m5 == None:
        m5 = band_error[band]['m5']
    
    x = 10**(0.4*(m-m5))
    sigma_rand = np.sqrt((0.04 - band_error[band]['gamma'])*x + band_error[band]['gamma']*x**2)
    sigma_1 = np.sqrt(sigma_sys**2 + sigma_rand**2) 

    return sigma_1
    
def sigmaN(m,band='r',m5=None,N=1):
    return 0.005 + sigma1(m,band)/N**0.5

def f_mass_radius_to_logg(M,R): 
    
    G = astropy.constants.G
    
    return np.log10((G*M/(R)**2).to(units.cm/units.s**2).value)

def f_mass_logg_to_R(M,logg): 
    
    G = astropy.constants.G
    
    return np.sqrt(G*M/(10**logg*units.cm/units.s**2)).to(units.R_sun)

def f_tau(T,wavlength):
    
    hckb = (astropy.constants.h*astropy.constants.c/astropy.constants.k_B)
    
    beta = (T<8000*units.K)*0.08 + (T>8000*units.K)*0.25 # Needs to be check # PHOEBE defines beta : 4*beta!!!!!
    
    num = beta*(hckb/(wavlength*T))
    
    den = 1-np.exp(-(hckb/(wavlength*T)))
    
    return num/den

def f_blackbody_fluxOnStarSurface(T, wavelength):

    # Planck's Law formula
    numerator = (2 * astropy.constants.h * astropy.constants.c**2) / (wavelength**5)
    
    exponent = (astropy.constants.h * astropy.constants.c) / (wavelength * astropy.constants.k_B * T)
    denominator = (np.exp(exponent) - 1)

    # Flux = pi*radiance
    return np.pi*(numerator/denominator)
    
def f_binary_separation(Period, M1, M2):
    
    M_total = (M1 + M2)
    G = astropy.constants.G
    
    a_cubed = (G * M_total * Period**2) / (4 * np.pi**2)
    a = a_cubed**(1 / 3)

    return a

def f_binary_K(Period, M1, M2):
    
    a = f_binary_separation(Period, M1, M2)

    K = a*2*np.pi/Period

    return K
    
def f_lambda(T1, T2, wavelength):
    
    hckb = (astropy.constants.h*astropy.constants.c/astropy.constants.k_B)
    term2 = np.exp(hckb / (wavelength * T2)) - 1
    term1 = np.exp(hckb / (wavelength * T1)) - 1
    
    return (T2 / T1)**4 * (term2 / term1)
    
lldc = Table(fits.open('/home/gadaman1/data_znadia1/01_Projects/01_WD/07_LSST_DWD/LinearLimbDarkeningCf_2013G.fits')[1].data)
t_lldc = lldc[lldc['Filt']=='g']
interp_lldc = scipy.interpolate.LinearNDInterpolator(np.transpose([t_lldc['Teff'].data,t_lldc['log_g_'].data]),\
                                       np.array(t_lldc['a']))

def f_AEllipsoidal(M1, M2, R1, T1, Period, inclination, waveband=4640*units.Angstrom):
    
    G = astropy.constants.G
    
    tau1 = f_tau(T1,waveband) # Needs to be checked
    
    mu1 = interp_lldc((T1.to(units.K).value,f_mass_radius_to_logg(M1,R1))) # Needs to checked
    if np.isnan(mu1):
        mu1 = 0.5
        
    numerator = 3*np.pi**2*(15+mu1)*(1+tau1)*M2*R1**3*np.sin(inclination)**2
    denominator = 5*(Period)**2*(3-mu1)*G*M1*(M1+M2)

    A_ellipsoidal = numerator/denominator
    
    return A_ellipsoidal.decompose()

def f_ADoppler(M1, M2, R1, R2, T1, T2, Period, inclination, waveband=4640*units.Angstrom):

    K1 = f_binary_K(Period, M1, M2)*np.sin(inclination)
    K2 = f_binary_K(Period, M2, M1)*np.sin(inclination)
    
    hckb = (astropy.constants.h*astropy.constants.c/astropy.constants.k_B)
    
    x1 = hckb/(waveband*T1)
    x2 = hckb/(waveband*T2)
    
    alpha_1_prime = (x1 * np.exp(x1)) / (np.exp(x1) - 1)
    alpha_2_prime = (x2 * np.exp(x2)) / (np.exp(x2) - 1)

    luminosity_1 = (R1**2*f_blackbody_fluxOnStarSurface(T1,waveband))
    luminosity_2 = (R2**2*f_blackbody_fluxOnStarSurface(T2,waveband))
    
    numerator = K1*alpha_1_prime*luminosity_1 - K2*alpha_2_prime*luminosity_2
    denominator = luminosity_1 + luminosity_2

    A_Doppler = np.abs((1/astropy.constants.c)*numerator/denominator)
    
    return A_Doppler.decompose()

def f_ARelfection(M1, M2, R1, R2, T1, T2, Period, inclination, waveband=4640*units.Angstrom):
    
    f_lambda_value = f_lambda(T1,T2,waveband)
    a = f_binary_separation(Period,M1,M2)
    
    # Terms for the second star:
    term_R2_1 = 24 * (R2 / a)**2
    term_R2_2 = 27 * np.pi * (R2 / a)**3
    term_R2_3 = 2 * (R2 / a)**2 * np.sin(inclination)**2
    
    # Terms for the first star:
    term_R1_1 = 24 * (R1 / a)**2
    term_R1_2 = 27 * np.pi * (R1 / a)**3
    term_R1_3 = 2 * (R1 / a)**2 * np.sin(inclination)**2
    
    #mag_factor = (R2**2*T2**4)/(R1**2*T1**4) # 10**(-0.4 * delta_m)
    
    luminosity_1 = (R1**2*f_blackbody_fluxOnStarSurface(T1,waveband))
    luminosity_2 = (R2**2*f_blackbody_fluxOnStarSurface(T2,waveband))
    mag_factor = luminosity_2/luminosity_1
    
    # Combine all terms:
    numerator = (term_R2_1 + term_R2_2 + term_R2_3)/f_lambda_value + \
                    mag_factor * (term_R1_1 + term_R1_2 + term_R1_3) * f_lambda_value
    denominator = 1 + mag_factor
    
    AReflection = np.abs(-(2 / (144 * np.pi)) * np.sin(inclination)**2 * numerator / denominator)
    return AReflection.decompose()

def f_sinusoidalFlux(t,A,Period,phase): 
    return A*np.sin(2*np.pi*t/Period*units.radian + phase*units.radian)

def f_sinusoidalFluxError(magnitude,band='r',m5=24): 
    return sigma1(magnitude,band,m5)
    
def lightcurve_SimulationAndRecover_plotting(timeObs, fluxObs, e_fluxObs, bandObs,\
                                            ell_lightcurve_folded, ell_periodogram, Period, model_ell_lightcurve_folded):
    #######################################
    # Plotting
    #######################################
    
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5),tight_layout=True)
    
    # Plot Lightcurve
    
    model_ell_lightcurve_folded.bin(bins=lightcurve_bins).plot(ax=ax[0])
    ell_lightcurve_folded.bin(bins=lightcurve_bins).errorbar(color='r',marker='o',ax=ax[0])
    ell_lightcurve_folded.errorbar(alpha=0.1,marker='o',c='grey',lw=2,ax=ax[0])
    ax[0].set_ylim(np.percentile(fluxObs,1),np.percentile(fluxObs,99))
    
    # Plot periodogram
    
    ell_periodogram.plot(view='period',scale='log',ax=ax[1])
    ax[1].set_yscale('linear')
    _ymin,_ymax = ax[1].set_ylim()
    _xmin,_xmax = ax[1].set_xlim()
    #ax[1].vlines(Period.to(ell_periodogram.period_at_max_power.unit).value,_ymin,_ymax,ls=':',label=f'Period = {Period.to(units.hour):.2f} \
    #                    \nBest Fit = {ell_periodogram.period_at_max_power.to(units.hour):.2f} ',color='r')
    
    #ax[1].hlines(y=ell_periodogram.max_power,xmax=_xmax,xmin=Period.to(ell_periodogram.period_at_max_power.unit).value,\
    #             ls='--',lw=2,label=f'Period = {Period.to(units.hour):.2f} \
    #                    \nBest Fit = {ell_periodogram.period_at_max_power.to(units.hour):.2f} ',color='r')

    temp_period = Period.to(ell_periodogram.period_at_max_power.unit).value
    
    ax[1].arrow(_xmax, ell_periodogram.max_power*0.75, temp_period-_xmax+10**(np.log10(temp_period)+np.log10(_xmax/_xmin)/20), 0, \
                head_width=(_ymax - _ymin)/40, \
                head_length=10**(np.log10(temp_period)+np.log10(_xmax/_xmin)/20-0.05), fc='red', ec='red',\
               label=f'Period = {Period.to(units.hour):.2f} \
                        \nBest Fit = {ell_periodogram.period_at_max_power.to(units.hour):.2f} ')

    ax[1].legend(fontsize=15, loc='upper right')
    
    return None

def validatePeriod(ell_periodogram,Period):
    return max(np.abs(ell_periodogram.period[(ell_periodogram.max_power-ell_periodogram.power)<2] - Period)) < Period*0.1

def combined_astropy(timeObs, fluxObs, e_fluxObs, model_fluxObs, bandObs, Period, \
                     maximum_period, minimum_period, plot):
    
    #######################################
    # LombScargle astropy Periodogram
    #######################################
    
    model = LombScargle(t=timeObs, y=fluxObs, dy=e_fluxObs)
    try:
        frequency,power = model.autopower(minimum_frequency=1/maximum_period.to(units.hour),\
                                          maximum_frequency=1/minimum_period.to(units.hour),\
                                          method="fastnifty")

        falseAlarmProbab = model.false_alarm_probability(power.max(),\
                                                         minimum_frequency=1/maximum_period.to(units.hour),\
                                                         maximum_frequency=1/minimum_period.to(units.hour))

        ls_period = (frequency[np.argmax(power)]**-1).value
    except:
        ls_period = 0
        falseAlarmProbab = 1
        
    return ls_period,falseAlarmProbab

def lightcurve_SimulationAndRecover(stellar_params,\
                        source_coordinate,\
                        observation_filters,\
                        source_magnitude,\
                        maximum_period,\
                        minimum_period,\
                        random_seed=1,\
                        nterms=1,\
                        model='combined_astropy',\
                        plot=False):

    np.random.seed(random_seed)

    M1,M2,R1,R2,T1,T2,Period,inclination = stellar_params
    
    #######################################
    # Select LSST observations
    #######################################
    
    observations = df_table[np.array((source_coordinate.separation(df_table['field_coordinates'])).value)<3.5/2]['observationStartMJD','fiveSigmaDepth','filter']
    
    if len(observations) > 0:
        observations['observationStartMJD'] = observations['observationStartMJD'] - min(observations['observationStartMJD'])

        #######################################
        # Select observation bands
        #######################################

        observation_index = np.argwhere([observation_filter in observation_filters for observation_filter in observations['filter']])
        observation_index = observation_index[:,0]

        #######################################
        # Simulate Observations
        #######################################

        timeObs = (observations['observationStartMJD'][observation_index]*units.day).to(units.hour)
        timeObs = timeObs - timeObs.min()

        bandObs = observations['filter'][observation_index]

        variationAmplitude = [f_ADoppler(M1,M2,R1,R2,T1,T2,Period,inclination,band_wavelengths[band]) for band in bandObs]
        
        phase = np.random.uniform(0,2*np.pi)
        
        model_fluxObs = f_sinusoidalFlux(timeObs,variationAmplitude,Period,phase)
        e_fluxObs = np.array([f_sinusoidalFluxError(source_magnitude[band],band,m5) for m5,band in observations['fiveSigmaDepth','filter'][observation_index]])
        fluxObs = np.random.normal(model_fluxObs,scale=e_fluxObs)

        if model == 'combined_astropy':
            ls_period,falseAlarmProbab = combined_astropy(timeObs, fluxObs, e_fluxObs, model_fluxObs, bandObs, Period, \
                                          maximum_period, minimum_period, plot)
            return ls_period,falseAlarmProbab

    else:
        return -1,1
    
def parallelize_simulation(params,plot=False,random_seeds=[1],output='brief'):
    M1,M2,R1,R2,T1,T2,Period,inclination,source_magnitude,source_coordinate,observation_filters,N_seeds = params

    random_seeds = range(N_seeds)
    
    ls_periods,falseAlarmProbabs = [],[]
    for random_seed in random_seeds:
        ls_period,falseAlarmProbab = lightcurve_SimulationAndRecover(stellar_params=[M1,M2,R1,R2,T1,T2,Period,inclination],\
                                                            source_coordinate=source_coordinate,\
                                                            observation_filters=observation_filters,\
                                                            source_magnitude=source_magnitude,\
                                                            maximum_period=50*units.hour,\
                                                            minimum_period=10*units.minute,\
                                                            random_seed=random_seed,model='combined_astropy',\
                                                            plot=plot)
        ls_periods.append(ls_period)
        falseAlarmProbabs.append(falseAlarmProbab)
        
    return [ls_periods,falseAlarmProbabs]
    
###########################################

dataset = Table.read('/home/gadaman1/01_Research/01_Projects/01_WD/08_LSST_DWD/01_SeBa/01_results/GalacticDWDs.ecsv')

N_seeds = 10
dataset.add_column(np.zeros((len(dataset),N_seeds)),name='beaming_period')
dataset.add_column(np.ones((len(dataset),N_seeds)),name='beaming_falseAlarmProbab')

recoverability = []
recoverability_index = []

for ii_dwd in tqdm(range(len(dataset))):
    dwd = dataset[ii_dwd]
    Period = dwd['period_hour']*units.hour
    if Period < 50*units.hour and 17 < dwd['r'] and dwd['r'] < 22:
        M1 = dwd['M1']*units.M_sun
        M2 = dwd['M2']*units.M_sun
        R1 = dwd['R1']*units.R_sun
        R2 = dwd['R2']*units.R_sun
        T1 = 10**dwd['logTeff1']*units.K
        T2 = 10**dwd['logTeff2']*units.K
        inclination = dwd['inclination']*units.degree
        source_magnitude = {'u':dwd['u'],'g':dwd['g'],'r':dwd['r'],'i':dwd['i'],'z':dwd['z'],'y':dwd['y']}

        source_RA = dwd['RA_ICRS']*units.degree
        source_DEC = dwd['DEC_ICRS']*units.degree

        source_coordinate = SkyCoord(source_RA,source_DEC,frame='icrs')

        observation_filters = list('ugrizy')

        params = [M1,M2,R1,R2,T1,T2,Period,inclination,source_magnitude,source_coordinate,observation_filters,N_seeds]

        recoverability.append(params)
        recoverability_index.append(ii_dwd)
    
###########################################
print('Starting Analysis')

num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) 

print(f'Size = {len(recoverability)} , Expected Time = {6*(len(recoverability)/61)*(24/num_cores)*(N_seeds/1):0.0f} seconds')
###########################################


print(f'Number of cores = {num_cores}')

initialTime = time.time()

if __name__ == '__main__':
    with multiprocessing.Pool(processes=num_cores) as pool:
        detection = pool.map(parallelize_simulation, recoverability)
    
finalTime = time.time()

print('End of Analysis')
print(f'Total Time = {finalTime-initialTime:0.0f}')

###########################################

detection = np.array(detection)

for ii in range(len(detection)):
    dataset['beaming_period'][recoverability_index[ii]] = detection[ii][0]
    dataset['beaming_falseAlarmProbab'][recoverability_index[ii]] = detection[ii][1]
    
dataset.write('/home/gadaman1/01_Research/01_Projects/01_WD/08_LSST_DWD/03_results/GalacticDWDBeaming.ecsv',overwrite=True)

print(len(dataset[np.sum(dataset['beaming_falseAlarmProbab']<0.05,axis=1)==N_seeds]),\
      len(dataset[np.sum(dataset['beaming_falseAlarmProbab']<0.003,axis=1)==N_seeds]))

###########################################

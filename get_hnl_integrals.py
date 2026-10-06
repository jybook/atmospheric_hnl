import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
from scipy.integrate import quad
import scipy.integrate as integrate

import nuflux
import functools
from multiprocessing import Pool, cpu_count
import vegas
import math

seconds = 10*365*24*60*60 #seconds in ten years of live time
R = 6357 #polar radius of earth in km
IC_depth = 1.950 #depth from surface to center of IceCube in km


def nutau_flux(E, cos_theta):
    '''
    Inputs:
    E, the D meson energy in GeV
    cos_theta, cosine of the zenith angle of the incoming D
    Output:
    differential flux for that energy and zenith angle
    
    We use nuflux (https://github.com/icecube/nuflux/blob/main/docs/overview.rst) 
    with H3a_SIBYLL23C_pr (https://arxiv.org/pdf/1806.04140)
    to get the prompt tau neutrino flux
    '''
    
    flux = nuflux.makeFlux('H3a_SIBYLL23C_pr')
    nu_type=nuflux.NuTau
    nu_energy=E # in GeV
    nu_cos_zenith = cos_theta
    flux_val = flux.getFlux(nu_type,nu_energy,nu_cos_zenith)
    
    return flux_val

def write_flux_dict(E_range, cos_theta_range):
    '''
    Run the flux calculation once for each energy and costheta, and then be done.
    '''
    flux_dict = pd.DataFrame({"E":[], "cos_theta": [], "flux": []})
    counter = 0
    for E in E_range:
        for cos_theta in cos_theta_range:
            flux_dict.loc[counter]=([E, cos_theta, nutau_flux(E, cos_theta)])
            counter += 1
    return flux_dict            

def surface_to_detector(cos_theta):
    '''
    Zenith is defined at the center of the detector, so we calculate distance from the center 
    of the detector, and then subtract 0.5km
    '''
    d = IC_depth

    neg_b = -2*(R-d)*cos_theta
    sqrt_b2_minus_4ac = np.sqrt(4*(R-d)**2*cos_theta**2 + 4*(2*R*d + d**2))
    two_a = 2
    
    distance = (neg_b + sqrt_b2_minus_4ac)/two_a # (neg_b - sqrt_b2_minus_4ac)/two_a is negative
    return distance - 0.5

def travel_distance (cos_zenith):
    '''
    From nuflux documentation:
    "The returned flux represents the flux incident on the surface of the earth, 
    no attempt is made to include oscillations or interaction losses while traveling through the Earth."
    
    However, we see a peak in HNL parent production at around 10-20km above earth's surface. 
    Therefore, assuming HNLs are produced as soon as the parent decays we need to take this distance traveled
    into account (probablisitcally), as well as the distance from the surface to the detector.
    
    According to (https://arxiv.org/pdf/1910.12839), parent fluxes are maximal at 15.4 km above the surface, 
    and drop off above energies of 10^10 GeV.
    At 10^10GeV, the mean decay distances (in the lab frame) are of the order cm or mm, and they scale with energy. 
    This is negligible compared to the height above earth's surface at which they are produced (and indeed, 
    are beyond the resolution of our detector), so we ignore this distance.
    
    We therefore approximate HNL travel distance as 15.4 + distance travelled through the earth.
    '''

    return 15.4 + surface_to_detector(cos_zenith)

def lifetime (mHNL, U2):
    '''
    HNL rest lifetime, as a function of its mass (GeV) and mixing with the tau neutrino.
    Using decay width calculation from HNLDecayEnergy via LeptonInjector
    '''
    # Appriximation, previously used
    # return 1*(10**(-6)/U2)*(0.1/mHNL)**5 
    
    # More precise method:

    lifetime_natural_units = 1/(U2*FullWidth(mHNL)) # GeV^-1
    GeV_to_sec = 6.582e-25
    lifetime_sec = lifetime_natural_units*GeV_to_sec 
    return lifetime_sec

def survival_probability (E, cos_zenith, mHNL=0.01, U=0.001):
    '''
    The chance an HNL gets from production to the edge of the detector
    '''
    c_km = 3*10**5 #km
    c = 3*10**8
    
    travel_time = travel_distance(cos_zenith)/c_km #in the lab frame
    lorentz_gamma = E/mHNL#E/(mHNL*c)
    
    exponent = -travel_time/(lorentz_gamma*lifetime(mHNL, U))

    if (np.exp(exponent) > 0).any:
        return np.exp(exponent)
    return 1 + exponent

def decay_probability (E, mHNL=0.1, U=0.001):
    '''
    The chance an HNL decays at some point in the detector
    1 - chance HNL travels all the way through the detector
    '''
    c_km = 3*10**5 #km
    c = 3*10**8
    
    travel_time = 1/c_km #in the lab frame
    lorentz_gamma = E/mHNL #E/(mHNL*c)
    survival_prob = np.exp(-travel_time/(lorentz_gamma*lifetime(mHNL, U)))
    return 1-survival_prob

def cross_sectional_area():
    '''
    Gives the cross sectional area relevant to our flux and decay calculations
    '''
    area = 10**10 #1 km^2 in units of cm^2
    return area

##############################################################
#  Putting it all together
##############################################################

# Get the fluxes we'll need for the following integrals
# flux_dict = write_flux_dict(e_range, ctheta_range)

def total_events_predicted(m, U, integration_vars):
    '''
    Events predicted to reach the detector in 10 years
    '''
    seconds = 10*365*24*60*60 # seconds in ten years of live time
    cos_zenith, logE  = integration_vars
    E = 10**logE
    consts = 2*np.pi*seconds*U #Ie, not zenith dependent   
    A = cross_sectional_area()
    
#     flux = flux_dict[flux_dict['E']==E][flux_dict['cos_theta']==cos_zenith]['flux']
    
    survival_prob = survival_probability (E, cos_zenith, m, U)
    jacobian = E*np.log(10)
    
    return nutau_flux(E, cos_zenith)*survival_prob*consts*A*jacobian

def total_events_detected(m, U, integration_vars):
    seconds = 10*365*24*60*60 # seconds in ten years of live time
    cos_zenith, logE  = integration_vars
    E = 10**logE
    jacobian = E*np.log(10)
    consts = 2*np.pi*seconds*U #Ie, not zenith dependent   
    
    A = cross_sectional_area()
    
    survival_prob = survival_probability (E, cos_zenith, m, U)
    decay_prob = decay_probability(E, m, U)
    
    return nutau_flux(E, cos_zenith)*survival_prob*consts*decay_prob*A*jacobian

##############################################################
#  Incorporate branching fractions
##############################################################
from icecube.LeptonInjector import HNLDecayEnergy
FullWidth = np.vectorize( lambda x: HNLDecayEnergy.FullWidth(x) )
gamma_nu_nu_nu_overload = np.vectorize( lambda x: HNLDecayEnergy.gamma_nu_nu_nu_overload(x) )

def fraction_visible(mHNL):
    '''
    Calculates the fractions of decays at this mass which produce ANY visible energy
    This does NOT account for the likelihood of seeing an event given a specific detector energy threshhold
    '''
    if type(mHNL)== int:
        mHNL = float(mHNL)
    full_decay_width = FullWidth(mHNL)
    decay_00 = gamma_nu_nu_nu_overload(mHNL) # decays to 3 neutrinos produce no visible energy
    return 1-(decay_00/full_decay_width)


##############################################################
#  Do the integrals
##############################################################
n_points = 100
cos_z_range = [-1, 1]
E_range = [1, 5] # in log space, Gev
m_range = np.logspace(-2, 0.3, n_points)
U_range = np.logspace(-6.5, 0, n_points)


# Initialize Vegas Integrator
integ = vegas.Integrator([cos_z_range, E_range])

# Function to calculate events
def compute_event(params):
    mass, mixing = params
#     part_func = functools.partial(total_events_predicted, mass, mixing)
#     result_incident = integ(part_func, nitn=10, neval=1000)
    
    part_func_detected = functools.partial(total_events_detected, mass, mixing)
    result_detected = integ(part_func_detected, nitn=10, neval=1000)
#     return mass, mixing, result_incident.mean, result_detected.mean
    return mass, mixing, result_detected.mean

# Create a list of parameters for all combinations
param_list = [(mass, mixing) for mass in m_range for mixing in U_range]


###########################
# Do the thing   ##########
###########################



# Parallel processing
if __name__ == "__main__":
    # Use all available CPUs or specify the number
    num_workers = cpu_count()
    with Pool(num_workers) as pool:
        results = pool.map(compute_event, param_list)

    # Convert results to DataFrame
#     events_predicted = pd.DataFrame(results, columns=["m", "U", "num_incident", "num_detected"])
    events_predicted = pd.DataFrame(results, columns=["m", "U", "num_detected"])
    events_predicted["num_visible"] = events_predicted['num_detected']*fraction_visible(events_predicted['m'])

    # Save to CSV
    events_predicted.to_csv('predicted_events_with_branching_1_2_25.csv', index=False)
    
    ###############
    # Make plots  #
    ###############

    X, Y = np.meshgrid(m_range, U_range, indexing='ij')

    z = np.asarray(events_predicted["num_detected"])
    z = z.reshape((n_points, -1))
    
    visible_events = events_predicted['num_visible']
    visible_events = visible_events.reshape((n_points, -1))

    # Automatic selection of levels works; setting the
# log locator tells contourf to use a log scale:

    fig, ax = plt.subplots()
    
    lev_exp = np.arange(-20., 9)
    levs = np.power(10, lev_exp)
    cs = ax.contourf(X, Y, visible_events, levs, norm=colors.LogNorm())
    
    cbar = fig.colorbar(cs)
    
    CS2 = plt.contour(cs, levels=[10, 100, 1000], colors='r')
    cbar.add_lines(CS2)
    
    ax.clabel(CS2, inline=True, fontsize=10)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('HNL mass (GeV)')
    ax.set_ylabel(r"$U_{\tau}^2")
    ax.set_xlim(min(m_range), max(m_range))
    ax.set_ylim(min(u_range), max(u_range))
    fig.savefig("visible_events_limits.png")
# plt.legend()
    
#     #Isoparticle contour for incident HNLs
#     X, Y = np.meshgrid(m_range, U_range)
#     n_events = np.array_split(events_predicted["num_incident"], len(m_range))

#     fig, ax = plt.subplots()
#     CS = ax.contour(X,Y,n_events)

#     ax.clabel(CS, inline=True, fontsize=10)
#     ax.set_title('Incident events in 10 years')
#     ax.set_xlabel("HNL mass (GeV)")
#     ax.set_ylabel("Mixing with tau neutrino (U^2)")
#     ax.set_xscale ("log")
#     ax.set_yscale("log")
    
#     fig.savefig("incident3.png")
    
#     #Isoparticle contour for detected HNLs
#     n_events_det = np.array_split(events_predicted["num_detected"], len(m_range))

#     fig1, ax1 = plt.subplots()
#     CS1 = ax1.contour(X,Y,n_events_det)

#     ax1.clabel(CS1, inline=True, fontsize=10)
#     ax1.set_title('Events detected in 10 years')
#     ax1.set_xlabel("HNL mass (GeV)")
#     ax1.set_ylabel("Mixing with tau neutrino (U^2)")
#     ax1.set_xscale ("log")
#     ax1.set_yscale("log")
    
#     fig1.savefig("detected3.png")

    
    
    

    

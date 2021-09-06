######
# Numba implementation of optimal control problem
######

import numpy as np
import sys

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output

from tqdm import tqdm  # Den här måste jag pipa
from multiprocessing import Pool  # Den här behöver jag eventuellt pipa

import multiprocessing as mp

from functools import partial
from run_all_parallel_eta_rho import experiment_allCBA, experiment_allCBA2

# ---kod som har med parallelliseringen att göra--------------

# Dessa två funktioner måste ligga i en egen .py-fil. Finns med här så jag vet vad som händer i Pool nedan
# def magic_function(f):
#     return f+10
#
# def process_frame(f):
#     # changed your logic here as I couldn't repro it
#     return f, magic_function(f)
#
# frames_list = [0, 1, 2, 3, 4, 5, 6]
#
# max_pool = 5
#
# with Pool(max_pool) as p:  #Jag använder 5 processer i det här exemplet.
#     pool_outputs = list(  #pool_outputs samlar in alla resultat. Är det så jag ska göra? Tror inte det, mina resultat skrivs ut till en enskild fil i varje steg
#         tqdm( #Används för att få fram en progress-bar
#             p.imap(process_frame,  #Här skapas processerna och process_frame anges som huvudfunktion
#                    frames_list),   #Här läggs in de värden som ska processas av funktionen.
#             total=len(frames_list) #Detta är väl information som tqdm behöver för att veta hur långt den kommit
#         )
#     )
# Om det är så enkelt som det verkar så ska jag anpassa funktionen experiment_allCBA() nedan så att den tar emot
# listor med exempelvis SSPi. Det skulle ge 5 parallella processer. Om jag delar upp diskonteringsräntorna istället
# så skulle jag kunna fördela ut allt på flera processer. Börja enkelt med att bara ändra på en variabel!
# print(pool_outputs)
# new_dict = dict(pool_outputs)

# print("dict:", new_dict)
# ------------------------------------------------------------

# rhos = np.array([0.0001,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])/100.
# etas = np.array([0, 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5])+0.00001
rhos = np.array([0.7, 1.7]) / 100.
etas = np.array([0.7, 1.7]) + 0.00001
SSPlist = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
PRTP_elasmu_n = [(r, e) for r in rhos for e in etas]
SSP_PRTP_elasmu_n = [(s, r, e) for s in SSPlist for r in rhos for e in etas]
print(SSP_PRTP_elasmu_n)
#SSPlist = list(enumerate(['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']))
nprocs = 40 #mp.cpu_count()
print(f"Number of CPU cores: {nprocs}")
#pool = mp.Pool(processes=nprocs)
max_pool_n = 5


def parallel_runs(frames_list, max_pool=5):
    #if __name__ == '__main__':
    #    __spec__ = None
    if __name__ == '__main__':
        with Pool(max_pool) as p:  # Jag använder 5 processer i det här exemplet.
            print(SSPlist)
            print(PRTP_elasmu_n)
            ssp_run = partial(experiment_allCBA, beta=2.0, PRTP_elasmu=PRTP_elasmu_n, withInertia=True, filename='allCBA',
                              TCRE_values=[0.62], minEmissions=-20)
            #tqdm(  # Används för att få fram en progress-bar
            pool_outputs = list(
                p.imap(ssp_run, frames_list)#,
                    # Här skapas processerna och process_frame anges som huvudfunktion
                    #   #Här läggs in de värden som ska processas av funktionen. Som det är definierat nu så är det SSP som ska ändras!
            #    total=len(frames_list)  # Detta är väl information som tqdm behöver för att veta hur långt den kommit
            #)
            )


def parallel_runs2(frames_list,max_pool=40):
    #if __name__ == '__main__':
    #    __spec__ = None
    if __name__ == '__main__':
        disc_run = partial(experiment_allCBA2, beta=2.0, withInertia=True,
                           filename='allCBA', TCRE_values=[0.62], minEmissions=-20)
        with Pool(max_pool) as p:  # Jag använder 5 processer i det här exemplet.
        #with pool as p:  # Jag använder 5 processer i det här exemplet.
#            print(SSPlist)
#            print(PRTP_elasmu_n)
#            ssp_run = partial(experiment_allCBA, beta=2.0, PRTP_elasmu=PRTP_elasmu_n, withInertia=True, filename='allCBA',
#                              TCRE_values=[0.62], minEmissions=-20)
            #tqdm(  # Används för att få fram en progress-bar
        #pool_outputs = list(
            pool_outputs = list(
                p.map_async(disc_run, frames_list).get()#,
                    # Här skapas processerna och process_frame anges som huvudfunktion
                    #   #Här läggs in de värden som ska processas av funktionen. Som det är definierat nu så är det SSP som ska ändras!
            #    total=len(frames_list)  # Detta är väl information som tqdm behöver för att veta hur långt den kommit
            #)
            )
            print(pool_outputs)
#####
# Run experiments #Det kommer bara bli ett experiement här! allCBA
#####

#parallel_runs(SSPlist, 5)
parallel_runs2(SSP_PRTP_elasmu_n)#, 20)

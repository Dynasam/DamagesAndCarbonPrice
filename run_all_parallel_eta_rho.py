######
# Numba implementation of optimal control problem
######

import numpy as np
import sys

from carbontaxdamages.defaultparams import Params
from carbontaxdamages.run import full_run_structured, export_output
from datetime import datetime
from tqdm import tqdm   #Den här måste jag pipa
from multiprocess import Pool #Den här behöver jag eventuellt pipa

import multiprocessing

from functools import partial

#---kod som har med parallelliseringen att göra--------------

#Dessa två funktioner måste ligga i en egen .py-fil. Finns med här så jag vet vad som händer i Pool nedan
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
#Om det är så enkelt som det verkar så ska jag anpassa funktionen experiment_allCBA() nedan så att den tar emot
#listor med exempelvis SSPi. Det skulle ge 5 parallella processer. Om jag delar upp diskonteringsräntorna istället
#så skulle jag kunna fördela ut allt på flera processer. Börja enkelt med att bara ändra på en variabel!
#print(pool_outputs)
#new_dict = dict(pool_outputs)

#print("dict:", new_dict)
#------------------------------------------------------------

experiments = {}

#####
# Define experiments
#####

#rhos = np.array([0.0001,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])/100.
#etas = np.array([0, 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5])+0.00001
# rhos = np.array([0.5,1.5])/100.
# etas = np.array([1.0,1.5])+0.00001
# PRTP_elasmu = [(r,e) for r in rhos for e in etas]
# SSPlist = ['SSP1', 'SSP1', 'SSP3', 'SSP4', 'SSP5']
# max_pool = 5

####################### Cost benefit runs ###########################

def experiment_allCBA(
	SSPlist = ["SSP1"],
    beta=2.0, 
#    PRTP_elasmu=[(0.001, 1.001), (0.015, 1.001), (0.03, 1.001)],
    PRTP_elasmu=[(0.001, 1.001)], 
    withInertia=True, 
    filename='allCBA',
#    TCRE_values=[0.42, 0.62, 0.82],
    TCRE_values=[0.62],
    minEmissions=-20
):
#    rhos = np.array([0.0001,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])/100.
#    etas = np.array([0, 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5])+0.00001
#    PRTP_elasmu = [(r,e) for r in rhos for e in etas]
    print(PRTP_elasmu)
    print("Running experiment 'allCBA'")
#    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
    SSP = SSPlist
#    for SSP in SSPlist: #['SSP1', 'SSP3', 'SSP4', 'SSP5']:
#    for SSP in ['SSP2']:
#        for damage in ['damageDICE', 'damageHowardTotal', 'damageBurkeWithLag']:
    for damage in ['damageHowardTotal']:
        for TCRE in np.array(TCRE_values) * 1e-3:
            # for cost_level in ['p05', 'p50', 'p95']:
            for cost_level in ['p50']:
                for PRTP, elasmu in PRTP_elasmu:
                    print(SSP, damage, TCRE, cost_level,elasmu,PRTP)
                    #print(PRTP)
                    #print(elasmu)
                    p_values_max_rel = 0.6
                    output = full_run_structured(Params(
                        damage=damage, beta=beta,
                        K_values_num=15 if withInertia else 25, CE_values_num=800 if withInertia else 1500,
                        p_values_num=600 if withInertia else 1000, p_values_max_rel=p_values_max_rel,
                        E_values_num=25 if withInertia else 2, maxReductParam=2.2 if withInertia else 500,
                        cost_level=cost_level, SSP=SSP, carbonbudget=0, r=PRTP,
                        elasmu=elasmu,
                        TCRE=TCRE,
                        minEmissions=minEmissions,
                        T=180, t_values_num=int(180 / 5) + 1,
                        useCalibratedGamma=True,
                        runname=f"Experiment {filename} %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions nobudget 2020 beta %beta elasmu %elasmu carbint %maxReductParam",
                        shortname="CBA %SSP %damage %TCRE %cost_level %r"
                    ))
                    export_output(output, plot=False)

#Nästa funktion tar en tuple med ssp, PRTP och elasmu för att loopa över
def experiment_allCBA2(
        # Omskrivning för parallell-körning. När jag använder partial måste min iterativa variabel ligga först
        SSP_PRTP_elasmu=[("SSP1", 0.001, 1.001)],
        #    SSPlist = ["SSP1","SSP2"],
        beta=2.0,
        #    PRTP_elasmu=[(0.001, 1.001), (0.015, 1.001), (0.03, 1.001)],
        withInertia=True,
        filename='allCBA',
        #    TCRE_values=[0.42, 0.62, 0.82],
        TCRE_values=[0.62],
        minEmissions=-20
):
    #    rhos = np.array([0.0001,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])/100.
    #    etas = np.array([0, 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5])+0.00001
    #    PRTP_elasmu = [(r,e) for r in rhos for e in etas]
    #    print(PRTP_elasmu)
    starttime = datetime.now()
    starting_time = starttime.strftime("%H:%M:%S")
    print("Running experiment 'allCBA'")
    #    for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
    #    print(SSPlist)
    elasmu = SSP_PRTP_elasmu[2]
    PRTP = SSP_PRTP_elasmu[1]
    SSP = SSP_PRTP_elasmu[0]
    print(SSP_PRTP_elasmu)
    print(PRTP)
    print(elasmu)
    print(starting_time)
    #   SSP = SSPlist
    #    for SSP in SSPlist: #['SSP1', 'SSP3', 'SSP4', 'SSP5']:
    #    for SSP in ['SSP2']:
    #        for damage in ['damageDICE', 'damageHowardTotal', 'damageBurkeWithLag']:
    for damage in ['damageHowardTotal']:
        for TCRE in np.array(TCRE_values) * 1e-3:
            # for cost_level in ['p05', 'p50', 'p95']:
            for cost_level in ['p50']:
                # for PRTP, elasmu in PRTP_elasmu:
                # for SSP in SSPlist: #['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
                print(SSP, damage, TCRE, cost_level, elasmu, PRTP)
                # print(PRTP)
                # print(elasmu)
                p_values_max_rel = 0.6
                output = full_run_structured(Params(
                    damage=damage, beta=beta,
                    K_values_num=15 if withInertia else 25, CE_values_num=800 if withInertia else 1500,
                    p_values_num=600 if withInertia else 1000, p_values_max_rel=p_values_max_rel,
                    E_values_num=25 if withInertia else 2, maxReductParam=2.2 if withInertia else 500,
                    cost_level=cost_level, SSP=SSP, carbonbudget=0, r=PRTP,
                    elasmu=elasmu,
                    TCRE=TCRE,
                    minEmissions=minEmissions,
                    T=180, t_values_num=int(180 / 5) + 1,
                    useCalibratedGamma=True,
                    runname=f"Experiment {filename} %SSP %damage TCRE %TCRE cost %cost_level r=%r %minEmissions nobudget 2020 beta %beta elasmu %elasmu carbint %maxReductParam",
                    shortname="CBA %SSP %damage %TCRE %cost_level %r"
                ))
                export_output(output, plot=False)
    endtime = datetime.now()
    current_time =endtime.strftime("%H:%M:%S")
    return(f"Sim och tid: {SSP} {damage} {TCRE} {cost_level} {elasmu} {PRTP} {starting_time} {current_time}")

# def parallel_runs(frames_list=SSPlist, max_pool=max_pool):
#     with Pool(max_pool) as p:  #Jag använder 5 processer i det här exemplet.
#         tqdm( #Används för att få fram en progress-bar
# 			ssp_run = partial(experiment_allCBA,beta=2.0, PRTP_elasmu=[(0.001, 1.001)], withInertia=True,filename='allCBA', TCRE_values=[0.62],minEmissions=-20)
# 			p.imap(ssp_run,  #Här skapas processerna och process_frame anges som huvudfunktion
#                 frames_list),   #Här läggs in de värden som ska processas av funktionen. Som det är definierat nu så är det SSP som ska ändras!
# 			total=len(frames_list) #Detta är väl information som tqdm behöver för att veta hur långt den kommit
# 		)
#     #)
#
#
#
# #####
# # Run experiments #Det kommer bara bli ett experiement här! allCBA
# #####
#
# parallel_runs(SSPlist,5)
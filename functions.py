# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:34:15 2022

@author: Lisa
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:09:36 2021

@author: Lisa
"""

# Initialisation

import pandas as pd
import numpy as np
import os
from scipy import optimize as opt
import matplotlib.pyplot as plt
import copy
import math
import plotly.graph_objects as go
import pickle
import plotly.express as px
from itertools import combinations_with_replacement
from itertools import permutations


#Class definition


class impacts_class:
    # Store and organize the impact values for a given calculated impact
    def __init__(self, 
                 nom, #str, the name of the impact
                 methode, #str, the name of the calculation method from which the impact comes
                 valeur_totale, #float, the value of the impact
                 valeur_phase, #pd.DataFrame, index = total value, impact and credit value, columns=phases (sum is close to total value, not exactly equal because of inaccuracy) 
                 valeur_credit #pd.DataFrame, values = positive impact value and negative impact value (sum is close to total value, not exactly equal because of inaccuracy)
                 ):
        self.name=nom
        self.method=methode
        self.valeur_totale=valeur_totale
        self.valeur_phase=valeur_phase
        self.valeur_credit=valeur_credit
        
        

class scenario_class:
    def __init__(self, name, optimized_impact, method, constraints, factors, impacts):
        self.name=name #str, scenario's name
        self.optimized_impact=optimized_impact #str, name of the optimized impact (several if multi-objective optimization)
        self.method=method #str, name of the calculation method from which the optimized impact comes
        self.constraints=constraints #str, information about the chosen constraints
        self.factors=factors #pd.DataFrame, columns = process' names, values = multiplicative factors for each process to reach the minimization of the optimized impact
        self.impacts=impacts #list of impact_class, all impacts linked to the scenario
        
    def all_impacts_parameter(self, parameter):        
        list_of_impact_parameter=[]
        
        if parameter=="name":
            for i in range(len(self.impacts)):
                list_of_impact_parameter.append(self.impacts[i].name)
        
        if parameter=="valeur_totale":
            for i in range(len(self.impacts)):
                list_of_impact_parameter.append(self.impacts[i].valeur_totale)
        
        if parameter=="valeur_rel":
            for i in range(len(self.impacts)):
                list_of_impact_parameter.append(self.impacts[i].valeur_phase[0].iloc[-1, 0])
            
        return(list_of_impact_parameter)
    
    def to_pickle(self, pickle_file):
        # Store data into an external pickle file
        # pickle_file : str, name and path to the save file
        # varable_to_pickle : any data you want to store
    
        # Its important to use binary mode
        file = open(pickle_file, 'ab')
        pickle.dump(self, file)					
        file.close()
            

################################################################

# Construction of the work file


def search_impacts(main_data_file='D:/Matières/Double diplôme Montreal/CIRAIG/Projet de maîtrise/Code/Source_files/main_LV.xlsx',
                   file_impacts='D:/Matières/Double diplôme Montreal/CIRAIG/Projet de maîtrise/Code/Source_files/Impacts_unitaires.xlsx',
                   file_to_write='D:/Matières/Double diplôme Montreal/CIRAIG/Projet de maîtrise/Code/Source_files/impacts.xlsx',
                   Impacts_first_column='P1 : Retention basin',
                   Impacts_separation_column="Pa2 : Natural rubber avoided production",
                   Impacts_last_column='market for sand | sand | Cutoff, S - RoW',
                   Impacts_IWP_Mid_first_line="Climate change, long term (kg CO2 eq (long))",
                   Impacts_IWP_Mid_last_line="Water scarcity (m3 world-eq)",
                   Impacts_IWP_End_first_line="Ecosystem quality | Climate change, ecosystem quality, long term",
                   Impacts_IWP_End_last_line="Water availability, terrestrial ecosystem (PDF.m2.yr)",
                   Impact_EF_3_0_first_line="EF v3.0 | acidification | accumulated exceedance (ae)",
                   Impact_EF_3_0_last_line="EF v3.0 | water use | user deprivation potential (deprivation-weighted water consumption)",
                   database="EV3.8",
                   open_file="n"):
   
    
    #Initialisation
    print("\n Fermer le fichier \n")
    print(file_to_write)
    print("\n s'il est ouvert. \n La procédure prend quelques secondes.")

    #Vérification if the database exists
    
    while (database!="EV3.6") and (database!="EV3.8"):
        input("La base de donnée n'est pas valide. Veuillez entrer EV3.6 ou EV3.8")

    #Step 0 : Récupérer les impacts déjà entrés
    data=pd.read_excel(main_data_file,header=0,index_col=1).loc[:,Impacts_first_column:Impacts_last_column]
    impacts_IWP_Midpoint=pd.DataFrame(index=data.loc[Impacts_IWP_Mid_first_line:Impacts_IWP_Mid_last_line,:].index)
    impacts_IWP_Endpoint=pd.DataFrame(index=data.loc[Impacts_IWP_End_first_line:Impacts_IWP_End_last_line,:].index)
    impacts_EF_3_0=pd.DataFrame(index=data.loc[Impact_EF_3_0_first_line:Impact_EF_3_0_last_line,:].index)
    
    # Step 1 : Lire le fichier contenant les indexs
    index=data.loc[database,Impacts_first_column:Impacts_last_column].fillna(0)


    # Step 2 : Transposer le tableau des impacts
    raw=pd.read_excel(file_impacts,database,header=0, index_col=0)
    raw_transposed=raw.transpose()

    # Step 3 : Remplir les colonnes une par une
    
    # Récupérer la localisation des impacts midpoint, endpoint et EF3.0
    Impacts_IWP_Mid_first_line_loc=raw_transposed.index.get_loc(Impacts_IWP_Mid_first_line)
    Impacts_IWP_Mid_last_line_loc=raw_transposed.index.get_loc(Impacts_IWP_Mid_last_line)
    Impacts_IWP_End_first_line_loc=raw_transposed.index.get_loc(Impacts_IWP_End_first_line)
    Impacts_IWP_End_last_line_loc=raw_transposed.index.get_loc(Impacts_IWP_End_last_line)
    Impact_EF_3_0_first_line_loc=raw_transposed.index.get_loc(Impact_EF_3_0_first_line)
    Impact_EF_3_0_last_line_loc=raw_transposed.index.get_loc(Impact_EF_3_0_last_line)
    
    # Récupérer les bonnes colonnes dans les bons tableaux

    for i in range(index.shape[0]):
        if (index[i]==0):
            column=data.iloc[:,i].name

            new_column=data.loc[Impacts_IWP_Mid_first_line:Impacts_IWP_Mid_last_line,column].fillna(0)
            impacts_IWP_Midpoint=pd.concat([impacts_IWP_Midpoint,new_column],axis=1)

            new_column=data.loc[Impacts_IWP_End_first_line:Impacts_IWP_End_last_line,column].fillna(0)
            impacts_IWP_Endpoint=pd.concat([impacts_IWP_Endpoint,new_column],axis=1)

            new_column=data.loc[Impact_EF_3_0_first_line:Impact_EF_3_0_last_line,column].fillna(0)
            impacts_EF_3_0=pd.concat([impacts_EF_3_0,new_column],axis=1)

        else :
            for j in range(raw_transposed.shape[1]):
                if (index[i]==raw_transposed.columns[j]):

                    new_column=raw_transposed.iloc[Impacts_IWP_Mid_first_line_loc:Impacts_IWP_Mid_last_line_loc+1,j]
                    new_column.index=impacts_IWP_Midpoint.index
                    impacts_IWP_Midpoint=pd.concat([impacts_IWP_Midpoint,new_column],axis=1)

                    new_column=raw_transposed.iloc[Impacts_IWP_End_first_line_loc:Impacts_IWP_End_last_line_loc+1,j]
                    new_column.index=impacts_IWP_Endpoint.index
                    impacts_IWP_Endpoint=pd.concat([impacts_IWP_Endpoint,new_column],axis=1)
                    
                    new_column=raw_transposed.iloc[Impact_EF_3_0_first_line_loc:Impact_EF_3_0_last_line_loc+1,j]
                    new_column.index=impacts_EF_3_0.index
                    impacts_EF_3_0=pd.concat([impacts_EF_3_0,new_column],axis=1)


    # Step 4 : Ecrire dans un fichier contenant les impacts
    with pd.ExcelWriter(file_to_write) as writer: 
        impacts_IWP_Midpoint.to_excel(writer,"IWP Impacts Midpoint")
        impacts_IWP_Endpoint.to_excel(writer,'IWP Impacts Endpoint')
        impacts_EF_3_0.to_excel(writer,"EF 3.0 Impacts")
    
    # Step 5 : Ouvir le fichier obtenu
    if open_file=="y":
        os.startfile(file_to_write)


def data_initialisation(main_data_file='D:/Matières/Double diplôme Montreal/CIRAIG/Documents de travail/main.xlsx',
                        file_impacts='D:/Matières/Double diplôme Montreal/CIRAIG/Documents de travail/impacts.xlsx',
                        file_to_write='D:/Matières/Double diplôme Montreal/CIRAIG/Documents de travail/data.xlsx',
                        A_first_column='P1 : Bassin de rétention',
                        A_last_column='market for sand | sand | Cutoff, S - RoW',
                        A_first_line='Utilisation de pneus (kg) (UF)',
                        A_last_line='Sable (kg)',
                        demand_first_line='Utilisation de pneus (kg) (UF)',
                        demand_last_line='Sable (kg)',
                        demand_first_column='Demande finale',
                        demand_last_column='Demande finale',
                        bounds_first_line='lb',
                        bounds_last_line='ub',
                        bounds_first_column='P1 : Bassin de rétention',
                        bounds_last_column='market for sand | sand | Cutoff, S - RoW',
                        phases_first_line='P.Production',
                        phases_last_line='P.End-of-life',
                        open_file="n"):

    print("\nLancer cette fonction après avoir formaté les données d'impacts dans ")
    print(file_impacts)
    print("\nFermer le fichier ")
    print(file_to_write)
    print("s'il est ouvert.\n")
    
    #Lecture du fichier
    raw=pd.read_excel(main_data_file,index_col=1)

    #On trouve A
    A=raw.loc[A_first_line:A_last_line,A_first_column:A_last_column]
    
    #On trouve demande
    demande=raw.loc[demand_first_line:demand_last_line,demand_first_column:demand_last_column]
    
    #On trouve les limites
    bounds=raw.loc[bounds_first_line:bounds_last_line,bounds_first_column:bounds_last_column]
    
    #On trouve les impacts
    impacts_IWP_Midpoint=pd.read_excel(file_impacts,"IWP Impacts Midpoint", index_col=0)
    impacts_IWP_Endpoint=pd.read_excel(file_impacts, 'IWP Impacts Endpoint', index_col=0)
    impacts_EF3_0=pd.read_excel(file_impacts, 'EF 3.0 Impacts', index_col=0)
    
    #On trouve les phases
    P=raw.loc[phases_first_line:phases_last_line,A_first_column:A_last_column]

    #Ecriture du fichier
    with pd.ExcelWriter(file_to_write) as writer: 
        A.to_excel(writer,"A")
        demande.to_excel(writer,'demande')
        bounds.to_excel(writer,'bounds')
        P.to_excel(writer,"phases")
        impacts_IWP_Midpoint.to_excel(writer,'IWP Impacts Midpoint')
        impacts_IWP_Endpoint.to_excel(writer,'IWP Impacts Endpoint')
        impacts_EF3_0.to_excel(writer,'EF 3.0 Impacts')

    # #Ouvir le fichier obtenu            
    xl=pd.ExcelFile(file_to_write)
    if open_file=="y":
        os.startfile(xl)
        
    return xl

################################################################
# Optimization


def optimisation(nom_fichier,method,impact_a_optimiser):
    
    ## Reading the data

    # First read the bounds
    bounds = pd.read_excel(nom_fichier, 'bounds', index_col=0)
    bounds = bounds.where(pd.notnull(bounds), None)
    # Tranform bounds Dataframe to a list of tuples
    bounds_list = [i for i in bounds.T.itertuples(index=False, name=None)]

    # Then read in the other matrices (A, final, and impacts)
    A = pd.read_excel(nom_fichier, 'A', index_col=0).fillna(0)
    impacts = pd.read_excel(nom_fichier, method, index_col=0).fillna(0)
    demande = pd.read_excel(nom_fichier, 'demande', index_col=0).iloc[:,-1].fillna(0)

    ##Optimisation
    res = opt.linprog(c=impacts.loc[impact_a_optimiser,:].squeeze().values,  
                  A_eq=A.values, 
                  b_eq=demande.values,
                  bounds=bounds_list, 
                  method='revised simplex')
    
    # Correction des erreurs d'arrondi
    for i in range (len(res['x'])):
              if (abs(res['x'][i])<1e-5):
                  res['x'][i]=0
    
    #L'optimisation renvoie 
    #les valeurs des coefficients multiplicateurs des process dans res['x'],
    #la valeur de l'impact total optimisé dans res['fun']
    
    opt_resultats_df=pd.DataFrame(data=res['x'],index=A.columns,columns=[impact_a_optimiser]).transpose()
    
    return opt_resultats_df


def independant_verif(matrix):
    import sympy
    matrix_val=matrix.values
    
    # Give the position and name of the dependant rows of the matrix
    ref_rows=list(range(0,matrix_val.shape[0]))
    _, inds_row = sympy.Matrix(matrix_val).T.rref()   # to check the rows you need to transpose!
    dep_row=set(ref_rows).difference(inds_row)
    
    if len(dep_row)!=0:
        print("Position of dependant rows: ", (dep_row))
        print("Name of dependant rows : \n")
        for i in range (len(dep_row)):
            print(matrix.index[list(dep_row)[i]])
    else:
        print("\n All rows are idependant")


    # Give the position and name of the dependant columns of the matrix
    ref_columns=list(range(0,matrix_val.shape[1]))
    _, inds_col = sympy.Matrix(matrix_val).rref()   # to check the rows you need to transpose!
    dep_col=set(ref_columns).difference(inds_col)
    
    if len(dep_col)!=0:
        print("\n Position of dependant columns :", (dep_col))
        print("\n Name of dependant columns :")
        for i in range (len(dep_col)):
            print(matrix.columns[list(dep_col)[i]])
    else:
        print("\n All columns are independant")

def calculate_impacts(resultat_optimisation,nom_fichier):
    # On crée une liste vide, qui contiendra tous les impacts
    list_impacts=[]
    
    # On calcule la matrice qui permet l'analyse de contribution par phase du cycle de vie
    A=pd.read_excel(nom_fichier, "A", index_col=0).fillna(0)
    D=pd.read_excel(nom_fichier, "demande", index_col=0).fillna(0)
    P=pd.read_excel(nom_fichier, "phases", index_col=0).fillna(0)
    S=pd.DataFrame(data=resultat_optimisation, index=A.columns, columns=["Values"])
    
    P_final=link_remaining_processes_to_phases(A, D, S, P)
    
    # On répète pour chaque méthode
    for j in range(4,len(nom_fichier.sheet_names)):
        
        # Méthode de calcul
        method=nom_fichier.sheet_names[j]
        
        # Calcul des impacts totaux
        impacts=pd.read_excel(nom_fichier, method, index_col=0).fillna(0)
        impacts_totaux=impacts@resultat_optimisation
        
        # Calcul des impacts positifs et évités
        # Attention, la somme des deux n'est pas exactement égale aux impacts totaux (erreurs d'arrondi)
        
        coeff_positifs=resultat_optimisation.copy()
        coeff_negatifs=resultat_optimisation.copy()
        
        for i in range(len(resultat_optimisation)):
            if (resultat_optimisation[i]>0):
                coeff_negatifs[i]=0
            if (resultat_optimisation[i]<0):
                coeff_positifs[i]=0
        
        impacts_positifs=impacts@coeff_positifs
        impacts_evites=impacts@coeff_negatifs
        
        # Calcul des impacts par phase du cycle de vie 
        
        impacts_scaled=impacts.dot(np.diag(resultat_optimisation))
        
        #impacts_per_phase=attributes_phase_impacts(impacts_scaled,P_final)
        
        # Création de la classe impacts : on range toutes les données pertinentes par catégorie d'impact, et on met dans une liste
        
        for i in range (len(impacts)):
            nom=impacts.index[i]
            methode=method
            valeur_totale=impacts_totaux[i]
            valeurs_par_credit=pd.DataFrame([impacts_positifs[i],impacts_evites[i]], index=["Impact positif","Impact évité"],columns=["Valeurs"])
            valeur_phase=attributes_phase_impacts(impacts_scaled.iloc[i,:],P_final),
            x=impacts_class(nom, methode, valeur_totale, valeur_phase, valeurs_par_credit)
            list_impacts.append(x)
            
            
    return list_impacts


def objective_choice(main_data_file):
    raw=pd.read_excel(main_data_file,index_col="Selected scenarios")
    list_objectives=[]
    for i in range (raw.shape[0]):
        if raw.index[i]=="X":
            list_objectives.append(raw.iloc[i,0])
    return(list_objectives)  



def link_remaining_processes_to_phases(A_matrix, D_matrix, S_matrix, P_matrix):
   #This function allows to distribute the processus d'arrière plan to the phases already identified
    # Calculation of A_scaled

    S_diag=pd.DataFrame(data=np.diag(S_matrix.iloc[:,0]), index=S_matrix.index, columns=S_matrix.index)
    A_scaled_full=A_matrix.dot(S_diag)

    # Removing useless lines and columns
    
    ## First, removing the useless lines  from A scaled and from the demand matrix

    A_scaled_lines_removed=A_scaled_full.copy(deep=True)
    D_mini=D_matrix.copy(deep=True)

    for i in range (A_scaled_full.shape[0]):
        test=0
        for j in range (A_scaled_full.shape[1]):
            if A_scaled_full.iloc[i,j]!=0:
                test=test+1
        if test==0:
            name_of_the_line=A_scaled_full.index[i]
            A_scaled_lines_removed=A_scaled_lines_removed.drop(name_of_the_line, axis=0)
            D_mini=D_mini.drop(name_of_the_line, axis=0)

    ## Removing the useless columns from A scaled and from the phase matrix

    A_scaled_mini=A_scaled_lines_removed.copy(deep=True)
    P_matrix_mini=P_matrix.copy(deep=True)

    for i in range (A_scaled_lines_removed.shape[1]):
        test=0
        for j in range (A_scaled_lines_removed.shape[0]):
            if A_scaled_lines_removed.iloc[j,i]!=0:
                test=test+1
        if test==0:
            name_of_the_column=A_scaled_lines_removed.columns[i]
            A_scaled_mini=A_scaled_mini.drop(name_of_the_column, axis=1)
            P_matrix_mini=P_matrix_mini.drop(name_of_the_column, axis=1)


    # total_suppressed_lines=A_scaled_full.shape[0]-A_scaled_mini.shape[0]
    # total_suppressed_columns=A_scaled_full.shape[1]-A_scaled_mini.shape[1]

    # print(str(total_suppressed_lines) + " lines and " + str(total_suppressed_columns) + " columns were suppressed by this operation.")

    # ## Dimensions check
    # if ((A_scaled_mini.shape[0]==D_mini.shape[0])and(A_scaled_mini.shape[1]==P_matrix_mini.shape[1])):
    #     print("Dimensions are OK")
    # else:
    #     print("An error occured in lines and columns suppressions. Dimensions are not respected ")
    #     return
    
    # Identification of the flows that remains undistributed
    
    ## Obtain the total inputs and outputs in the phases already identified
    P_vector=P_matrix_mini.sum(axis=0)  
    P_all_matrix=A_scaled_mini.dot(np.diag(P_vector))
    P_all_matrix["All phases"]=P_all_matrix.sum(axis=1)
    P_all_matrix
    
    ## Compare with final demand
    comp_with_D=[]
    for i in range (P_all_matrix.shape[0]):
        comp_with_D.append(D_matrix.iloc[i, 0]-P_all_matrix.iloc[i, -1])
    comp_with_D_df=pd.DataFrame(data=comp_with_D, index=A_scaled_mini.index, columns=["Comparison with final demand"])
    ##Identify the position of flows that must be distributed
    list_pos_flows=[]
    for i in range (comp_with_D_df.shape[0]):
        if abs(comp_with_D_df.iloc[i,0])>=(1E-6):
            list_pos_flows.append(i)
            
    # print("Flows that must be distributed identified at positions ")
    # print(list_pos_flows)

    # Distributing the remaning flows
    
    ##For each flow, verify where the value goes

    P_matrix_mini_final=P_matrix_mini.copy(deep=True)

    for i in range (len(list_pos_flows)):
        share_of_the_flow=0     # Will allow to calculate the actual share of the remaining flows to put into the phase matrix
        pos_flows=list_pos_flows[i]
        value_flow=comp_with_D_df.iloc[pos_flows,0]
        name_of_provider=""  #Will be helpful for later

    ## Identify the process that provides the flow
        provider_found=0
        for j in range(A_scaled_mini.shape[1]):
            test_provider=abs(A_scaled_mini.iloc[pos_flows,j]-value_flow)
            if test_provider<=(1E-5):
                name_of_provider=A_scaled_mini.columns[j]
                provider_found=provider_found+1
        
        if provider_found<1:
            name_of_provider="\033[1;31m no provider found"
        if provider_found>1:
            name_of_provider="\033[1;31m several providers found"
                
    ##Calcualtion of the total input and output of each phase and comparison with the value of the flow that must be distributed
        for k in range(P_matrix_mini.shape[0]):
            P_vector=P_matrix_mini.iloc[k,:]
            P_matrix_mini_per_phase=A_scaled_mini.dot(np.diag(P_vector))
            P_matrix_mini_per_phase[P_matrix_mini.index[k]]=P_matrix_mini_per_phase.sum(axis=1)
            P_input_value=P_matrix_mini_per_phase.iloc[pos_flows,-1]
            
    ##If the flow is consumed by a phase, identify how much and the name of the destination
            if P_input_value!=0:
                flow_destination=P_matrix_mini_per_phase.columns[-1]
                share_of_the_flow=abs(P_input_value/value_flow)

                if (name_of_provider=="\033[1;31m no provider found") or (name_of_provider=="\033[1;31m several providers found"):
                    print("\033[1;30m The flow named " + A_scaled_mini.index[pos_flows] + " produced by " + name_of_provider)
                    print("\033[1;30m is consumed by " + flow_destination + " with a share of " + str(share_of_the_flow) + "\n")
                
    ##Put the information in the phase matrix 
                P_matrix_mini_final.loc[flow_destination, name_of_provider]=share_of_the_flow

    # Re-scale the matrix to the original size

    P_matrix_final=P_matrix.copy(deep=True)
    for i in range (P_matrix_mini_final.shape[1]):
        column_name=P_matrix_mini_final.columns[i]
        for j in range(P_matrix.shape[1]):
            if column_name==P_matrix.columns[j]:
                P_matrix_final.loc[:,column_name]=P_matrix_mini_final.loc[:,column_name]

    return P_matrix_final


def attributes_phase_impacts(impact, P_matrix):
      
    # Creation of a df containing the impacts per phase (to be calculated per impact category)

    impact_per_phase_columns=["Total"]
    for i in range (P_matrix.shape[0]):
        impact_per_phase_columns.append(P_matrix.index[i])

    impact_per_phase_index=["Impact", "Credit", "Total"]
    
    # Creation of the matrix that will contain the values
    impact_per_phase_matrix=pd.DataFrame(data=None, index=impact_per_phase_index, columns=impact_per_phase_columns)

    
    # Total values calculations

    ##Calculate values
    total_total=sum(impact)

    total_impact=0
    total_credit=0
    for j in range (len(impact)):
        if impact[j]>0:
            total_impact=total_impact+impact[j]
        else:
            total_credit=total_credit+impact[j]

    ##Store values

    impact_per_phase_matrix.iloc[0,0]=total_impact
    impact_per_phase_matrix.iloc[1,0]=total_credit
    impact_per_phase_matrix.iloc[2,0]=total_total

    # Phase values calculations

    ##Calculate values
    for k in range (P_matrix.shape[0]):
        phase_name=P_matrix.index[k]
        phase_values=P_matrix.iloc[k,:]
        phase_values_diag=np.diag(phase_values)
        impact_phase=impact.dot(phase_values_diag)

        phase_total=sum(impact_phase)

        phase_impact=0
        phase_credit=0

        for l in range (len(impact_phase)):
            if impact_phase[l]>0:
                phase_impact=phase_impact+impact_phase[l]
            else:
                phase_credit=phase_credit+impact_phase[l]

        ##Store values

        impact_per_phase_matrix.iloc[0, impact_per_phase_matrix.columns.get_loc(phase_name)]=phase_impact
        impact_per_phase_matrix.iloc[1, impact_per_phase_matrix.columns.get_loc(phase_name)]=phase_credit
        impact_per_phase_matrix.iloc[2, impact_per_phase_matrix.columns.get_loc(phase_name)]=phase_total
    
    # Verify if sum = total
    
    for m in range(impact_per_phase_matrix.shape[0]):
        diff=impact_per_phase_matrix.iloc[m,0]
        for n in range(1,impact_per_phase_matrix.shape[1]):
            diff=diff-impact_per_phase_matrix.iloc[m,n]
        if diff>=0.1:
            print("Pour la catégorie d'impact " + str(impact.index[0]))
            print("Différence pour la ligne " + str(impact_per_phase_matrix.index[m]) + " : " + str(diff))
    
    return impact_per_phase_matrix

def optimisation_one_obj(nom_fichier,list_objectives,temporary_save_file, constraints="Standard", coment="" , open_file="n"):
    
# Renvoie une liste de tous les scénarios optimisés
# Chaque scénario contient le nom du scénario, l'impact optimisé, les facteurs multiplicatifs des processus et tous les impacts correspondants
# Les scénario sont des variables de type "scenario_class"
# Les résultats d'impact sont des variables de type "impacts_class"

    print("L'optimisation totale prend quelques secondes")
    
    # On crée la liste qui va contenir tous les résultats, sous la forme d'une liste de variables de classe "scenario"
    results=[]
    
    ## Reading the data

    # First read the bounds
    bounds = pd.read_excel(nom_fichier, 'bounds', index_col=0)
    bounds = bounds.where(pd.notnull(bounds), None)
    # Tranform bounds Dataframe to a list of tuples
    bounds_list = [i for i in bounds.T.itertuples(index=False, name=None)]

    # Then read the other matrices (A, final, and impacts)
    A = pd.read_excel(nom_fichier, 'A', index_col=0).fillna(0)
    demande = pd.read_excel(nom_fichier, 'demande', index_col=0).iloc[:,-1].fillna(0)

    # On crée le df qui va contenir la liste des facteurs, pour sauver en Excel
    scenario_total_factors=pd.DataFrame(index=A.columns)

    # Optimisation des impacts sélectionnés
    for i in range (3,len(nom_fichier.sheet_names)):
        
        method=nom_fichier.sheet_names[i]
        
        for j in range (pd.read_excel(nom_fichier,method).shape[0]):
            impacts = pd.read_excel(nom_fichier, method, index_col=0).fillna(0)
            
            for k in range (len(list_objectives)):
                if list_objectives[k]==impacts.index[j]:
                    impact_a_optimiser=pd.read_excel(nom_fichier,method).iloc[j,0]
                    
                    ##Optimisation
                    res = opt.linprog(c=impacts.loc[impact_a_optimiser,:].squeeze().values,  
                                  A_eq=A.values, 
                                  b_eq=demande.values,
                                  bounds=bounds_list, 
                                  method='revised simplex')
                        
                        # Correction des erreurs d'arrondi
                    for k in range (len(res['x'])):
                              if (abs(res['x'][k])<1e-5):
                                  res['x'][k]=0
                                  
                    #L'optimisation renvoie 
                    #les valeurs des coefficients multiplicateurs des process dans res['x'],
                    #la valeur de l'impact total optimisé dans res['fun']            
                    #avec res["x"] variable de type numpy nd array
                    
                    # Analyse         
                    
                    # Range tous les impacts du scénario optimisé dans une variable de classe impacts
                    list_impacts=calculate_impacts(res['x'],nom_fichier)
            
                    # Création d'une variable résumant toutes les données du scénario
                    scenario_name=coment + "opt. " + impact_a_optimiser
                    scenario_optimized_impact=impact_a_optimiser
                    scenario_method=method
                    scenario_constraints=constraints
                    scenario_factors=pd.DataFrame(data=res['x'],index=A.columns,columns=["Values"])
                    scenario_impacts=list_impacts
                    
                    scenario=scenario_class(scenario_name, scenario_optimized_impact, scenario_method, scenario_constraints, scenario_factors, scenario_impacts)
                    results.append(scenario)
                    
                    print(scenario.name + " has been optimized successfully")
                    
                    #Création d'un df pour sauver dans excel
                    scenario_factors_excel=pd.DataFrame(data=res['x'],index=A.columns,columns=[scenario_name])
                    scenario_total_factors=pd.concat([scenario_total_factors,scenario_factors_excel], axis=1)

    # Sauvegarde dans excel
    if temporary_save_file!=None:
        scenario_total_factors_transposed=scenario_total_factors.transpose()
        scenario_total_factors_transposed.to_excel(temporary_save_file)
        
        if open_file=="y":
            os.startfile(temporary_save_file)
                
    print('Optimization done !')
    
    return (results)


def optimisation_e_constraint(nom_fichier,list_objective_moo,list_of_e_constraint_values,temporary_save_file, constraints="Standard", coment="" , open_file="n"):
    
# Renvoie une liste de tous les scénarios optimisés
# Chaque scénario contient le nom du scénario, l'impact optimisé, les facteurs multiplicatifs des processus et tous les impacts correspondants
# Les scénario sont des variables de type "scenario_class"
# Les résultats d'impact sont des variables de type "impacts_class"

    print("L'optimisation totale prend quelques secondes")
    
    # On crée la liste qui va contenir tous les résultats, sous la forme d'une liste de variables de classe "scenario"
    results=[]
    
    ## Reading the data

    # First read the bounds
    bounds = pd.read_excel(nom_fichier, 'bounds', index_col=0)
    bounds = bounds.where(pd.notnull(bounds), None)
    # Tranform bounds Dataframe to a list of tuples
    bounds_list = [i for i in bounds.T.itertuples(index=False, name=None)]


    # Then read the other matrices (A, final, and impacts)
    A = pd.read_excel(nom_fichier, 'A', index_col=0).fillna(0)
    demande = pd.read_excel(nom_fichier, 'demande', index_col=0).iloc[:,-1].fillna(0)

    # On crée le df qui va contenir la liste des facteurs, pour sauver en Excel
    scenario_total_factors=pd.DataFrame(index=A.columns)

    # Find the main objective function data and the e-constraint objective function data
    for i in range (3,len(nom_fichier.sheet_names)):
        
        method=nom_fichier.sheet_names[i]
        
        for j in range (pd.read_excel(nom_fichier,method).shape[0]):
            impacts = pd.read_excel(nom_fichier, method, index_col=0).fillna(0)
            
            if list_objective_moo[0]==impacts.index[j]:
                impact_a_optimiser=pd.read_excel(nom_fichier,method).iloc[j,0]
                impact_a_optimiser_val=pd.read_excel(nom_fichier,method).iloc[j,1:].values
               
            for k in range (1,len(list_objective_moo)):
                if list_objective_moo[k]==impacts.index[j]:
                    new_line=pd.read_excel(nom_fichier,method).iloc[j,1:]
                    A_ub_df_t=pd.DataFrame(new_line.values, index=new_line.index)
                    A_ub_df=A_ub_df_t.transpose()                   

    # Optimisation pour chaque valeur de e-constraint 
    for l in range (len(list_of_e_constraint_values)):
        
        b_ub_val=list_of_e_constraint_values[l]

          
        ##Optimisation

        res = opt.linprog(c=impact_a_optimiser_val,  
                      A_ub=A_ub_df,
                      b_ub=b_ub_val,
                      A_eq=A.values, 
                      b_eq=demande.values,
                      bounds=bounds_list, 
                      method='revised simplex')

            
            # Correction des erreurs d'arrondi
        for m in range (len(res['x'])):
                  if (abs(res['x'][m])<1e-5):
                      res['x'][m]=0
                      
        #L'optimisation renvoie 
        #les valeurs des coefficients multiplicateurs des process dans res['x'],
        #la valeur de l'impact total optimisé dans res['fun']            
        #avec res["x"] variable de type numpy nd array
        
        # Analyse         
        
        # Range tous les impacts du scénario optimisé dans une variable de classe impacts
        list_impacts=calculate_impacts(res['x'],nom_fichier)
        
        # Création d'une variable résumant toutes les données du scénario
        scenario_name=coment + "opt. " + str(b_ub_val)
        scenario_optimized_impact=impact_a_optimiser
        scenario_method=method
        scenario_constraints=constraints
        scenario_factors=pd.DataFrame(data=res['x'],index=A.columns,columns=["Values"])
        scenario_impacts=list_impacts
        
        scenario=scenario_class(scenario_name, scenario_optimized_impact, scenario_method, scenario_constraints, scenario_factors, scenario_impacts)
        results.append(scenario)
        
        print(scenario.name + " has been optimized successfully")
        
        #Création d'un df pour sauver dans excel
        scenario_factors_excel=pd.DataFrame(data=res['x'],index=A.columns,columns=[scenario_name])
        scenario_total_factors=pd.concat([scenario_total_factors,scenario_factors_excel], axis=1)
                    
    # Sauvegarde dans excel
    if temporary_save_file!=None:
        scenario_total_factors_transposed=scenario_total_factors.transpose()
        scenario_total_factors_transposed.to_excel(temporary_save_file)
        
        if open_file=="y":
            os.startfile(temporary_save_file)
                
    print('Optimization done !')
    
    return (results)

def add_optimized_scenarios(nom_fichier,pre_calculated_data_file):
    # Calculate impacts and create scenario-type variables from a file containing the factors
    # The file must be an Excel file
    # The sheet names corresponds to the methods used
    # The columns names must correspond to the A column names
    # Each line contains first the optimised impact name and then the vector s

    # On crée la liste qui va contenir tous les résultats, sous la forme de variables de classe "scenario"
    results=[]

    #Verification if the processes names corresponds to A columns
    A_columns=pd.read_excel(nom_fichier, "A").columns
    pre_calculated_data_file=pd.ExcelFile(pre_calculated_data_file)    
    for i in range (len(pre_calculated_data_file.sheet_names)):
        method=pre_calculated_data_file.sheet_names[i]
        pre_calculated_data_columns=pd.read_excel(pre_calculated_data_file,method).columns
        if np.array_equal(A_columns,pre_calculated_data_columns)==False:
            print("Dans l'onglet " + method + "les noms des processus diffèrent. Veuillez vérifier que les processus sont les mêmes")

        #Trouver les listes de facteur et calculer les impacts associés
        for j in range (pd.read_excel(pre_calculated_data_file,method).shape[0]):
            impact_a_optimiser=pd.read_excel(pre_calculated_data_file,method,index_col=0).index[j]
            factors=pd.read_excel(pre_calculated_data_file,method).iloc[j,1:].values
            list_impacts=calculate_impacts(factors,nom_fichier)
        
            # Création d'une variable résumant toutes les données du scénario
            scenario_name="pc.scn. " + impact_a_optimiser
            scenario_optimized_impact=impact_a_optimiser
            scenario_method=method
            scenario_factors=factors
            scenario_impacts=list_impacts
            scenario=scenario_class(scenario_name, scenario_optimized_impact, scenario_method, scenario_factors, scenario_impacts)
            results.append(scenario)

    return (results)


def optimisation_to_norm(nom_fichier,list_objectives):
    
# Renvoie un df avec le nom de l'objectif, et les paramètres nécessaires pour le normaliser
# Le df contient en ligne les objectifs, en colonne les fi utopia et les fi max (respectivement le point idéal et le point de Nadir)
# Both are necessary to normalize fi, norm= (fi(x) - uto_fi)/nadir_fi-uto_fi)

    
    # On crée le df qui va contenir tous les résultats
    norm_factor_df=pd.DataFrame(index=list_objectives, columns=["ideal_f", "nadir_f"])
    
    # On crée la liste qui va contenir toutes les valeurs possibles des objectifs
    # Chaque mini liste à l'intérieur contient toutes les valeurs possibles d'un ibjectif
    possible_values=[[] for i in range(len(list_objectives))]
    
    # On optimise les 4 objectifs séparément
    ## Reading the data

    # First read the bounds
    bounds = pd.read_excel(nom_fichier, 'bounds', index_col=0)
    bounds = bounds.where(pd.notnull(bounds), None)
    # Tranform bounds Dataframe to a list of tuples
    bounds_list = [i for i in bounds.T.itertuples(index=False, name=None)]

    # Then read the other matrices (A, final, and impacts)
    A = pd.read_excel(nom_fichier, 'A', index_col=0).fillna(0)
    demande = pd.read_excel(nom_fichier, 'demande', index_col=0).iloc[:,-1].fillna(0)

    # On crée le df qui va contenir la liste des facteurs, pour sauver en Excel
    scenario_total_factors=pd.DataFrame(index=A.columns)

    # Optimisation des impacts sélectionnés
    for i in range (3,len(nom_fichier.sheet_names)):
        
        method=nom_fichier.sheet_names[i]
        
        for j in range (pd.read_excel(nom_fichier,method).shape[0]):
            impacts = pd.read_excel(nom_fichier, method, index_col=0).fillna(0)
            
            for k in range (len(list_objectives)):
                if list_objectives[k]==impacts.index[j]:
                    impact_a_optimiser=pd.read_excel(nom_fichier,method).iloc[j,0]
                    
                    ##Optimisation
                    res = opt.linprog(c=impacts.loc[impact_a_optimiser,:].squeeze().values,  
                                  A_eq=A.values, 
                                  b_eq=demande.values,
                                  bounds=bounds_list, 
                                  method='revised simplex')
                        
                        # Correction des erreurs d'arrondi
                    for l in range (len(res['x'])):
                              if (abs(res['x'][l])<1e-5):
                                  res['x'][l]=0
                                  
                    #L'optimisation renvoie 
                    #les valeurs des coefficients multiplicateurs des process dans res['x'],
                    #la valeur de l'impact total optimisé dans res['fun']            
                    #avec res["x"] variable de type numpy nd array
                    
                    ## Analyse
                    
                    print(impact_a_optimiser + " has been optimized successfully")

                    # Calculation of all impact values with this x
                    # On répète pour chaque méthode car il y a 3 feuilles d'impacts à explorer
                    for x in range(3,len(nom_fichier.sheet_names)):

                        # Méthode de calcul
                        method_fin=nom_fichier.sheet_names[x]

                        # Calcul des impacts totaux
                        impacts_fin=pd.read_excel(nom_fichier, method_fin, index_col=0).fillna(0)
                        impacts_totaux=impacts_fin@res['x']
                        
                        for y in range(len(list_objectives)):
                            for z in range(impacts_totaux.shape[0]):
                                if list_objectives[y]==impacts_totaux.index[z]:
                                    possible_values[y].append(impacts_totaux.values[z])
                    
    # uto_f and nadir_f calculation
    
    for m in range(len(possible_values)):
        ideal_f=min(possible_values[m])
        nadir_f=max(possible_values[m])
        norm_factor_df.iloc[m,0]=ideal_f
        norm_factor_df.iloc[m,1]=nadir_f     
        
    print('Normalization factors calculated !')
    
    return (norm_factor_df)


################################################################

# Values manipulation - multi-objective optimization

def add_new_impact_cat(list_scenario, new_impact_name, new_impact_method, sum_list):
    
    # Print information about the calculation
    print("Création de la nouvelle catégorie ")
    print(new_impact_name)
    
    print("\nAddition des valeurs des catégories d'impact")
    for j in range (len(sum_list)):
        k=sum_list[j]
        print(list_scenario[0].impacts[k].name)
    print("\n")

    for i in range(len(list_scenario)):
        # Calcul des valeurs totales et valeurs par crédit
        valeur_totale=0
        valeur_par_credit_posit=0
        valeur_par_credit_negat=0

        for j in range (len(sum_list)):
            k=sum_list[j]
            valeur_totale=valeur_totale+list_scenario[i].impacts[k].valeur_totale
            valeur_par_credit_posit=valeur_par_credit_posit+list_scenario[i].impacts[k].valeur_credit.iloc[0,0]
            valeur_par_credit_negat=valeur_par_credit_negat+list_scenario[i].impacts[k].valeur_credit.iloc[1,0]

        valeurs_par_credit=pd.DataFrame([valeur_par_credit_posit,valeur_par_credit_negat], index=["Impact positif","Impact évité"],columns=["Valeurs"])

        # Final impact type variable
        new_impact=impacts_class(new_impact_name, new_impact_method, valeur_totale, valeurs_par_credit)

        # Addition to the previous impact list
        list_scenario[i].impacts.append(new_impact)

    print("Catégorie ajoutée !\n")
    
    return(list_scenario)

def gen_weights_combinaisons(list_of_possible_weights, number_of_fuctions_to_be_weighted):
    # Allows to obtain all the possible combinaisons of (number_of_fuctions_to_be_weighted) numbers
    # from a defined list of possible weights (list_of_possible_weights)
    
    # To obtain all combinaisons which the sum is equal to 1

    L2=[i for i in combinations_with_replacement(list_of_possible_weights,number_of_fuctions_to_be_weighted)]
    L3=[]
    for i in range (len(L2)):
        minisum=0
        for j in range(len(L2[i])):
            minisum=minisum+L2[i][j]
        if minisum==1:
            L3.append(L2[i])
    
    # To obtain all the permutation for all the unique combinaisons obtained earlier

    L4=[]

    for i in range(len(L3)):
        perm=[i for i in permutations(L3[i])]
        for j in range(len(perm)):
            L4.append(perm[j])


    #Remove the duplicates

    L5=list(set(L4))

    # Classer par ordre croissant

    L6=sorted(L5, reverse=True)
    return(L6)


################################################################

#Analysis and graphs


def scenario_choice(scenarios_list,nb_comp=None):
    #Allows to select the scenarios the user wants to be compared, from the list of optimized scenarios
    # scenario_list is a list containing scenario_type variables
    # nbr_comp is a int containin the number of scenarios the user wants to be compared. If non, the user can choose with input
    
    #Creation of the empty list which will contain the final scenario selection

    #Initialisation
    scenarios_selection=[]
    test="n"    
    
    # #Choosing the number of scenarios compared
    if nb_comp==None:
        while test=="n":
            try:
                nb_comp=int(input("How many scenarios would you like to compare ?\n"))
                if (nb_comp>len(scenarios_list)):
                    print("The value must be under " + str(len(scenarios_list)))
                else:
                    test="y"
                
            except (ValueError):
                    print("Oops!  That was no valid number.  Try again...")

    #Choosing the scenarios
    for i in range (len(scenarios_list)):
        print(str(i) +" : " + str(scenarios_list[i].name))
    
    j=0
    while (j<nb_comp):
        k=int(input("Enter the number of the scenario chosen\n"))
        scenarios_selection.append(scenarios_list[k])
        j=j+1
        
    return(scenarios_selection)

def extract_impacts(scenario_list,value_chosen="valeur_totale",excel_file=None):
    # Allows the extraction of the impact values to be able to visualize and treat the data.
    # scenario_list list of scenario type variables
    # value_chosen str containing the data you want to extract. Must match the accepted str treated by the all_impacts parameter function
    # excel_file str containing the path for the storage. If none, no storage
    
    # Creation of the final df containing all the data
    final_df=pd.DataFrame(index=scenario_list[0].all_impacts_parameter("name"))
    for i in range (len(scenario_list)):
        final_df [scenario_list[i].name]=scenario_list[i].all_impacts_parameter("valeur_totale")
    
    if  (excel_file!=None):
        final_df.to_excel(excel_file)
    
    return final_df
    
def drawcamembert(values,names, title):
    
    # Gestion des couleurs
    # col=['#ff5722',
    #      '#ff9800',
    #      '#ffeb3b',
    #      '#cddc39',
    #      '#4caf50',
    #      '#4caf50',
    #      '#26a69a',
    #      '#26c6da',
    #      '#42a5f5',
    #      '#5c6bc0',
    #      '#7e57c2',
    #      '#ab47bc',
    #      '#ec407a',
    #      '#bc475e',
    #      '#b0bec5',
    #      '#b0bec5']
    
    col=['#08519c',
         '#3182bd',
         '#6baed6',
         '#9ecae1',
         '#c6dbef',
         '#810f7c',
         '#8856a7',
         '#de2d26',
         '#fc9272',
         '#31a354',
         '#a1d99b',
         '#e5f5e0',
         '#b0bec5',
         '#b0bec5',
         '#b0bec5',
         '#b0bec5']
    
    # Suppression des légendes pour les valeurs nulles (sinon surcharge visuelle)
    names_wt_useless=names.tolist()
    for i in range(len(names)):
        if values[i]==0:
            names_wt_useless[i]=" "
    
    
    # Création du graphique
    # plt.pie(values, labels=names,colors=col, autopct=lambda p : '{:.2f}%  ({:,.0f})'.format(p,p * sum(values)/100), labeldistance=1.15, wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' })
    plt.pie(values, labels=names_wt_useless,colors=col, autopct='%1.1f%%', pctdistance=0.85, labeldistance=1.15, wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' })
    
    # Add a title
    plt.title(title)
    
    print("\nLa quantité de matière totale du scénario vaut ")
    print(sum(values))
    #Show the graphic
    #plt.show();
    
    
def normalize_values(values_list):
    
    # values_list est une liste de liste de valeurs
    norm_values_list=copy.deepcopy(values_list)

    for i in range (len(values_list)):
        for j in range (len(values_list[0])):
            max_val=[]
            for k in range(len(values_list)):
                max_val.append(values_list[k][j])                    
                maw_val_pos =  [abs(ele) for ele in max_val]
            if max(maw_val_pos)==0:
                norm_values_list[i][j]=0
            else:
                norm_values_list[i][j]=values_list[i][j]/max(maw_val_pos)
    return(norm_values_list)

    

def draw_barplot(values_list,labels_list,xlabels,ylabel,title, colors):
    # value_list=list of list of all impacts values by scenario
    # labels_list=scenario names
    # xlabels=impact names
    # ylabel=name of the y axis
    # title=title of the graph
    # col=colors of the bars

    x = np.arange(len(xlabels))  # the label locations
    width = 0.7/len(labels_list)  # the width of the bars
    
    for i in range(len(values_list)):
        position= width*(i-(len(labels_list)-1)/2)
        plt.bar(x - position, values_list[i], width=width, label=labels_list[i],color=colors[i], edgecolor='black')
        
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.title(title,fontsize=60)
    plt.xticks(x,xlabels,fontsize=25)
    plt.ylabel(ylabel,fontsize=15)
    plt.xticks(rotation=0)    
    plt.yticks(fontsize=15)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize=15)
    
    #plt.show()    

def draw_stacked_barplot(list_list_data, label_inputs, label_outputs, title, fig_dim, color_palette, ylabel=None, xlabel=None, r_output_labels="n"):
    
    # Les données data doivent être présentées sous la forme d'une liste de listes
    # Chaque sous_liste contient les valeurs de tous les impacts relés à une phase ou un process ([impact process 1 sur CC short, sur CC long, etc])
    # label_input= list containing the labels for input data (the contributors to the final impact)
    # label_output= list containint the labels of the final total impacts
    
    width = 0.4       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots(figsize=fig_dim)
    
    bottom=0

    col=color_palette
    if r_output_labels=="y":
        ax.tick_params(labelrotation=90)
        
    # Obtain the cumulative sum of data to place the bottom correctly
    # From there it is a code portion from https://stackoverflow.com/questions/35979852/stacked-bar-charts-using-python-matplotlib-for-positive-and-negative-values
    data=np.array(list_list_data)
    
    # Take negative and positive data apart and cumulate
    def get_cumulated_array(data, **kwargs):
        cum = data.clip(**kwargs)
        cum = np.cumsum(cum, axis=0)
        d = np.zeros(np.shape(data))
        d[1:] = cum[:-1]
        return d  

    cumulated_data = get_cumulated_array(data, min=0)
    cumulated_data_neg = get_cumulated_array(data, max=0)

    # Re-merge negative and positive data.
    row_mask = (data<0)
    cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    data_stack = cumulated_data
    
    #my code againe from there
    
    for i in range(len(list_list_data)):
        bottom=data_stack[i]
        ax.bar(label_outputs, list_list_data[i], width, bottom=bottom, label=label_inputs[i], color=col[i])


    if ylabel!=None:
        ax.set_ylabel(ylabel)
        
    ax.set_title(title, loc='left', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1))

    plt.show()    
    

    
def draw_distribution(list_scenarios, bi=0, bs=13, nb_graph_per_line=2):
   
    # Comparaison de la répartition de la matière 
    
    # On définit les valeurs et les index du graphique
    
    ## On définit l'index
    index=list_scenarios[0].factors.index[bi:bs]
    
    ##On définitles valeurs
    values=[]
    
    for i in range(len(list_scenarios)):
        values.append(list_scenarios[i].factors[bi:bs].values.flatten())

    
    #On trace la fenêtre de graphiques
    plt.figure("Comparaison répartition", figsize=(10,math.floor((len(list_scenarios)+1)/2)*15))
    plt.gcf().subplots_adjust(left = 0.135, bottom = 0.2, right = 1.5,
                              top = 0.9, wspace = 0.5, hspace = 0.3)

    #Pour chaque scenario
    for i in range(len(list_scenarios)):
        for j in range(1,nb_graph_per_line):    
            plt.subplot(len(list_scenarios),nb_graph_per_line,i+j)
            drawcamembert(values[i], index, "Répartition de matière entre les voies de fin de vie,\n" + list_scenarios[i].name + "\n")
                
    plt.show()
    
def draw_comparison_distribution(list_scenarios, bi=0, bs=13, norm="n"):
    
    # Comparaison de la répartition de la matière 
    
    # On définit les valeurs et les index du graphique
    
    ## On définit l'index
    index=list_scenarios[0].factors.index[bi:bs]
    
    ##On définitles valeurs
    values=[]
    scenario_names=[]
    
    for i in range(len(list_scenarios)):
        values.append(list_scenarios[i].factors[bi:bs].values.flatten())
        scenario_names.append(list_scenarios[i].name)
        
    #On trace la fenêtre de graphiques
    plt.figure("Comparaison répartition", figsize=(math.floor(len(list_scenarios)/2)*10,10))

        
    #On normalise si souhaité
    if norm=="y":
        values_norm=normalize_values(values)
        values=values_norm
    
    # Graph drawing
    title=('Répartition de matière entre les voies de fin de vie \n' + list_scenarios[i].name + '\n Quantité [kg]')
    draw_barplot(values, scenario_names, index, "ylabel", title)

    plt.show()

def compare(values_1,values_2):
    diff_impacts=[]
    for i in range (len(values_1)):
        diff_impacts.append(((values_2[i]-values_1[i])/values_1[i])*100)
    return(diff_impacts)

def substract(values_1, values_2):
    diff_impacts=[]
    for i in range (len(values_1)):
        diff_impacts.append((values_2[i]-values_1[i]))
    return(diff_impacts)

def draw_comparison_impacts(list_scenarios, title, colors, bi=0,bs=3, norm="n", rel="n", ref=0 ):
    
    # Comparison of the impacts of each scenario
    
    # list_scenarios contains the list of selected secnarios for the comparison (list of scenario class variables)
    # bi is the number of the first impact of the analysis (in the order presented in the main matrix)
    # bs is the number of the last impact of the analysis (in the order presented in the main matrix)
    # rel is needed to be "y" to display relatve impacts
    # ref is the reference taken when displaying the relative impacts (the 0)
        
    # Index definition
    index_impacts=list_scenarios[0].all_impacts_parameter("name")[bi:bs]
    shorter_index_impacts=[]
    
    # To shorten the index if needed
    for i in range(len(index_impacts)):
        element=index_impacts[i].split('|')[-1]
        shorter_index_impacts.append(element)
    index_impacts=shorter_index_impacts
    
    # Values definition
    values_impact=[]
    index_scenario_names=[]
    for i in range(len(list_scenarios)):
        values_impact.append(list_scenarios[i].all_impacts_parameter("valeur_totale")[bi:bs])
        index_scenario_names.append(list_scenarios[i].name)

    # Normalization vs the biggest value in each impact category
    if norm == "y":
        values_impact_norm=normalize_values(values_impact)
        values_impact_fin=values_impact_norm
    else :
        values_impact_fin=values_impact
    
    
    # Display settings (change if there is a lot of data displayed)
    if (bs-bi)>15:
        # When a lot of data
        plt.figure(1,figsize=(25,15))
    
    if (bs-bi)<15:
        # When not that much of data
        plt.figure(1,figsize=(12,12))
    
    # Absolute values graph
    
    if rel!="y":
        ylabel=""
        draw_barplot(values_impact_fin,index_scenario_names,index_impacts,ylabel,title, colors)
    
    # Relative values graph

    
    if rel=="y":
        values_impact_rel=[]
        for i in range(len(list_scenarios)):
            imp_s1_sx=compare(values_impact[ref], values_impact[i])
            values_impact_rel.append(imp_s1_sx)
            
        ylabel="Différence relative entre les impacts [%]"
        draw_barplot(values_impact_rel,index_scenario_names,index_impacts,ylabel,title, colors)
        
    plt.show()
    
    # # Troisième graphique : Comparaison relative entre les répartitions de matière dans les voies de fin de vie
    # plt.subplot(3,1,3)

    # diff_s1_s2=compare_coeff(scenario_1.factors, scenario_2.factors)
    # title="Ecart relatif des répartitions de la matière dans les voies de fin de vie de " + name_sc_1 + " et " + name_sc_2
    # xlegend="Différence relative de répartition [%]"
    # drawbarplot(diff_s1_s2[0:15], diff_s1_s2.index[0:15], title, xlegend)
    
def scale_the_matrix(matrix,coeffs):
# Allow to multiply each column of the matrix by its corresponding coefficient contained in "coeffs"

    # Returned matrix initialisation
    scaled_matrix=pd.DataFrame(columns=matrix.columns, index=matrix.index).fillna(0)   # will contain the final matrix
    n_lines_matrix=matrix.shape[0]   # contains the number of lines of the matrix
    n_columns_matrix=matrix.shape[1]   #contains the number of columns of the matrix

    # Mathematical operation
    for i in range (n_columns_matrix):
        for j in range (n_lines_matrix):
            scaled_matrix.iloc[j,i]=matrix.iloc[j,i]*coeffs.iloc[i]

    return(scaled_matrix)

def matrix_to_sankey(matrix):
    
    # Create the links
    source_data=[]   # will contain the list of processes producing the flows
    target_data=[]   # will contain the list of processes consuming the flows
    value_data=[]    # will contain the list of flows sizes/values

    n_lines=matrix.shape[0]   # contains the number of lines of the matrix
    n_columns=matrix.shape[1]   #contains the number of columns of the matrix

    for i in range(n_columns):
        for j in range(n_lines):
            if matrix.iloc[j,i]>0:
                verification_variable=0   # allows to verify if the flow is consumed by another process, and not lost
                for k in range(n_columns):
                    if matrix.iloc[j,k]<0:
                        source_data.append(i)
                        target_data.append(k)
                        value_data.append(-matrix.iloc[j,k])
                        verification_variable=verification_variable+1
                if verification_variable==0:
                    print(matrix.index[j] + " is lost in darkness, except if this is the UF")
    
    # Draw the sankey                
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 20,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = matrix.columns,
          color = "light green"
        ),
        link = dict(
          source = source_data, # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = target_data,
          value = value_data
      ))])

    fig.update_layout(title_text="System flows", font_size=10)
    fig.show()


def scenario_list_to_df(scenario_list, transpose="n"):
    # Extract a df containing impact names and values for each scenario
    # Each column represents a scenario
    scenario_name=[]
    impact_names=[]
    impact_values=[]

    for i in range (len(scenario_list)):
        scenario_name.append([scenario_list[i].name])
        impact_names.append(scenario_list[i].all_impacts_parameter("name"))
        impact_values.append(scenario_list[i].all_impacts_parameter("valeur_totale"))
        new_line=pd.DataFrame(impact_values[i], index=impact_names[i], columns=scenario_name[i])
        if (i==0):
            impacts_df=new_line
        else:
            impacts_df=pd.concat([impacts_df,new_line], axis=1)
            
    # Possibility to transpose if the transpose variable contains "y", each line will represent a scenario
    if transpose=="y":
        impacts_df=impacts_df.transpose()

    return (impacts_df)

def draw_impact_comparison_parallel(scenario_list, list_col, column_color_ref, color_scale, title, automatic_color_settings="y"):
    # Draw a parallel coordinate graph from a list of scenarios
    # Each line of the df must represent a scenario
    
    # First set all the parameters
    dataframe=scenario_list_to_df(scenario_list, transpose="y")
    coordinates=[dataframe.columns.values[x] for x in list_col]
    
    # To rename specific axes
    short_names={'IWP Mid | Climate change, long term (kg CO2 eq (long))':'CC long',  # to change the names of the culumns if needed
                         'IWP Mid | Climate change, short term (kg CO2 eq (short))':'CC (kg CO2eq)',
                         'IWP Mid | Fossil and nuclear energy use (MJ deprived)':'FNEU (MJ deprived)',
                         'IWP Mid | Freshwater acidification (kg SO2 eq)':'FA',
                         'IWP Mid | Freshwater ecotoxicity (CTUe)':'FEco',
                         'IWP Mid | Freshwater eutrophication (kg PO4 P-lim eq)':'FEu',
                         'IWP Mid | Human toxicity cancer (CTUh)':'HTC',
                         'IWP Mid | Human toxicity non cancer (CTUh)':'HTNC',
                         'IWP Mid | Ionizing radiations (Bq C-14 eq)':'IR',
                         'IWP Mid | Land occupation, biodiversity (m2 arable land eq .yr)':'LOB',
                         'IWP Mid | Land transformation, biodiversity (m2 arable land eq)':'LTB',
                         'IWP Mid | Marine eutrophication (kg N N-lim eq)':'M eu',
                         'IWP Mid | Mineral resources use (kg deprived)':'MRU',
                         'IWP Mid | Ozone Layer Depletion (kg  CFC-11 eq)':'OLD',
                         'IWP Mid | Particulate matter formation (kg PM2.5 eq)':'PMF',
                         'IWP Mid | Photochemical oxidant formation (kg NMVOC eq)':'POF',
                         'IWP Mid | Terrestrial acidification (kg SO2 eq)':'TA',
                         'IWP Mid | Water scarcity (m3 world-eq)':'WS',
                         'IWP End | Ecosystem quality | Climate change, ecosystem quality, long term (PDF.m2.yr)':'EQ CC long',
                         'IWP End | Ecosystem quality | Climate change, ecosystem quality, short term (PDF.m2.yr)':'EQ CC short',
                         'IWP End | Ecosystem quality | Freshwater acidification (PDF.m2.yr)':'EQ FA',
                         'IWP End | Ecosystem quality | Freshwater ecotoxicity, long term (PDF.m2.yr)':'EQ FEco long',
                         'IWP End | Ecosystem quality | Freshwater ecotoxicity, short term (PDF.m2.yr)':'EQ FEco short',
                         'IWP End | Ecosystem quality | Freshwater eutrophication (PDF.m2.yr)':'EQ FEu',
                         'IWP End | Ecosystem quality | Ionizing radiation, ecosystem quality (PDF.m2.yr)':'EQ IR',
                         'IWP End | Ecosystem quality | Land occupation, biodiversity (PDF.m2.yr)':'EQ LOB',
                         'IWP End | Ecosystem quality | Land transformation, biodiversity (PDF.m2.yr)':'EQ LTB',
                         'IWP End | Ecosystem quality | Marine acidification, long term (PDF.m2.yr)':'EQ MA long',
                         'IWP End | Ecosystem quality | Marine acidification, short term (PDF.m2.yr)':'EQ MA short',
                         'IWP End | Ecosystem quality | Marine eutrophication (PDF.m2.yr)':'EQ ME',
                         'IWP End | Ecosystem quality | Terrestrial acidification (PDF.m2.yr)':'EQ TA',
                         'IWP End | Ecosystem quality | Thermally polluted water (PDF.m2.yr)':'EQ TPW',
                         'IWP End | Ecosystem quality | Water availability, freshwater ecosystem (PDF.m2.yr)':'EQ WAFE',
                         'IWP End | Ecosystem quality | Water availability, terrestrial ecosystem (PDF.m2.yr)':'EQ WATE',
                         'IWP End | Ecosystem quality |Ecosystem quality total (PDF.m2.yr)': 'EQ TOTAL',
                         'IWP End | Ecosystem quality | Ecosystem quality total red (PDF.m2.yr)' : 'EQ (PDF.m2.an)',
                         'IWP End |Human health | Climate change, human health, long term (DALY)':'HH CC long',
                         'IWP End | Human health | Climate change, human health, short term (DALY)':'HH CC short',
                         'IWP End | Human health | Human toxicity cancer, long term (DALY)':'HH HTC long',
                         'IWP End | Human health | Human toxicity cancer, short term (DALY)':'HH HTC short',
                         'IWP End | Human health | Human toxicity non-cancer, long term (DALY)':'HH HTNC long',
                         'IWP End | Human health | Human toxicity non-cancer, short term (DALY)':'HH HTNC short',
                         'IWP End | Human health | Ionizing radiation, human health (DALY)':'HH IR',
                         'IWP End | Human health | Ozone layer depletion (DALY)':'HH OLD',
                         'IWP End | Human health | Particulate matter formation (DALY)':'HH PMF',
                         'IWP End | Human health | Photochemical oxidant formation (DALY)':'HH POF',
                         'IWP End | Human health | Water availability, human health (DALY)':'HH WA',
                         'IWP End | Human health | Human health total (DALY)' : 'HH TOTAL',
                         'IWP End | Human health | Human health total red (DALY)' : 'HH (DALY)',
                         'EF 3.0 |acidification | accumulated exceedance (ae) (mol H+ eq)':'A',
                         'EF 3.0 | climate change | global warming potential (GWP100) (kg CO2 eq)':'CC',
                         'EF 3.0 | climate change: biogenic | global warming potential (GWP100) (kg CO2 eq)':'CC B',
                         'EF 3.0 | climate change: fossil,  global warming potential (GWP100) (kg CO2 eq)':'CC F',
                         'EF 3.0 | climate change: land use and land use change, global warming potential (GWP100) (kg CO2 eq)':'CC LU',
                         'EF 3.0 | ecotoxicity: freshwater, comparative toxic unit for ecosystems (CTUe)':'Eco F',
                         'EF 3.0 | ecotoxicity: freshwater, inorganic, comparative toxic unit for ecosystems (CTUe) ':'Eco F I',
                         'EF 3.0 | ecotoxicity: freshwater, metals, comparative toxic unit for ecosystems (CTUe) ':'Eco F M',
                         'EF 3.0 | ecotoxicity: freshwater, organics, comparative toxic unit for ecosystems (CTUe) ':'Eco F O',
                         'EF 3.0 | energy resources: non-renewable, abiotic depletion potential (ADP): fossil fuels (MJ)':'ERNR',
                         'EF 3.0 | eutrophication: freshwater,  fraction of nutrients reaching freshwater end compartment (P) (kg P eq)':'Eu F',
                         'EF 3.0 | eutrophication: marine, fraction of nutrients reaching marine end compartment (N) (kg N eq)':'Eu M',
                         'EF 3.0 | eutrophication: terrestrial, accumulated exceedance (AE) (mol N eq)':'Eu T',
                         'EF 3.0 | human toxicity: carcinogenic, comparative toxic unit for human (CTUh) ':'HTC',
                         'EF 3.0 | human toxicity: carcinogenic, inorganics, comparative toxic unit for human (CTUh) ':'HTC I',
                         'EF 3.0 | human toxicity: carcinogenic, metals, comparative toxic unit for human (CTUh) ':'HTC M',
                         'EF 3.0 | human toxicity: carcinogenic, organics, comparative toxic unit for human (CTUh) ':'HTC O',
                         'EF 3.0 | human toxicity: non-carcinogenic, comparative toxic unit for human (CTUh) ':'HTNC',
                         'EF 3.0 | human toxicity: non-carcinogenic, inorganics, comparative toxic unit for human (CTUh) ':'HTNC I',
                         'EF 3.0 | human toxicity: non-carcinogenic, metals, comparative toxic unit for human (CTUh) ':'HTNC M',
                         'EF 3.0 | human toxicity: non-carcinogenic, organics,  comparative toxic unit for human (CTUh) ':'HTNC O',
                         'EF 3.0 | ionising radiation: human health, human exposure efficiency relative to u235 (kBq U-235 eq)':'IR',
                         'EF 3.0 | land use, soil quality index (Pt)':'LU',
                         'EF 3.0 | material resources: metals/minerals, abiotic depletion potential (ADP): elements (ultimate reserves) (kg Sb eq)':'MR',
                         'EF 3.0 | ozone depletion, ozone depletion potential (ODP) (kg CFC11 eq)':'ODP',
                         'EF 3.0 | particulate matter formation, impact on human health (disease inc.)':'PMF',
                         'EF 3.0 | photochemical ozone formation: human health, tropospheric ozone concentration increase (kg NMVOC eq)':'POF',
                         'EF 3.0 | water use, user deprivation potential (deprivation-weighted water consumption) (m3 depriv.)':'WU'
                         }
    
    
    # Automatic color settings
    if automatic_color_settings=="y":
        
        col_values=[i for i in range (1, (len(scenario_list))+1)]
        new_column=pd.DataFrame(data=col_values, index=dataframe.index, columns=["Colors"])
        dataframe=pd.concat([dataframe, new_column], axis=1)

        column_color_ref="Colors"

        fig = px.parallel_coordinates(data_frame=dataframe, 
                                      dimensions=coordinates, 
                                      color=column_color_ref, 
                                      labels=short_names,
                                      color_continuous_scale=color_scale,
                                      title=title)
        
        # Hide the color scale that is useless in this case
        fig.update_layout(coloraxis_showscale=False, font_size=15)
    
    else :
        fig = px.parallel_coordinates(data_frame=dataframe, 
                              dimensions=coordinates, 
                              color=column_color_ref, 
                              labels=short_names,
                              color_continuous_scale=color_scale,
                              title=title)
    
    fig.show()

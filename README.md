# Optimization of the European end-of-life tire management system

This code is linked to the article : *doi will be put here*

Code created to optimize a life cycle assessment product system for environmental impacts when several processes can provide the same function. 
The optimization tool takes the useful files in the folder "source_files", stores temporary useful files in the folder "working_files" and stores results in the folder "results".

## Tool

functions.py : contains all the functions

single_objective_optimization_(SOO).ipynb allows to optimize one product system for one environmental impact

single_objective_optimization_loop_(SOOL).ipynb allows to easily optimize several product systems, each one for one environmental impact

multi-objective_optimization_weighted_sum_(MOOWS).ipynb needs to be used when the user wants to optimize for several objectives using the weighted sum method

multi-objective_optimization_e-constraint_(MOOeC).ipynb needs to be used when the user wants to optimize for several objectives using the weighted sum method

## Source files

user_file.xlsx : contains all the system products in a matrix form

impact_file.xlsx : contains unitary impacts for background processes 

*For privacy and copyright reasons, the user file and impact file provided here are given only as examples and do not contain any of the data used in the article.*



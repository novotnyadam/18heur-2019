#!/usr/bin/env python
# coding: utf-8

# # Genetic Optimization and Elitism
# ### Author: Adam Novotny, 18HEUR, 2019

# This Jupyter Notebook is concerned with **Genetic Optimization (GO)** and **Elitism** proposal demonstrated on Traveling Salesman Problem (TSP). 
# 
# ## Tasks
# * 1. To tune hyperparameters for **GO** on **``TSPGrid(3, 3)``**: to find optimal crossover, mutation and correction parameters in the first phase;  $N$, $M$, $T_1$ and $T_2$ parameters in the second phase,
# 
# * 2. To compare 'vanilla' **GO** and **Elitism**,
# 
# * 3. (Matej's proposal:) To introduce **Elitism** in combination with **Parasitism** and compare with results from 2.
# 
# ## Goals
# * 1. To improve Cauchy mutation performance by introducing Lévy mutation,
# 
# * 2. To improve **GO** by introducing **Elitism**.
# 

# ## Initialization

# In[1]:


# Import path to source directory (bit of a hack in Jupyter)
import sys
import os
pwd = get_ipython().run_line_magic('pwd', '')
sys.path.append(os.path.join(pwd, os.path.join('..', 'src')))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Import external librarires
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

import matplotlib
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# **``TSPGrid(3, 3)``** will be used for demonstration purposes.

# In[3]:


from objfun_tsp import TSPGrid
tsp = TSPGrid(3, 3)


# In[4]:


from heur_go import Crossover, UniformMultipoint, RandomCombination # crossover
from heur_go import GeneticOptimization # heuristics 
from heur_aux import Correction, MirrorCorrection, ExtensionCorrection # corrections 
from heur_aux import CauchyMutation, LevyMutation # mutations


# In[5]:


def rel(x):
    return len([n for n in x if n < np.inf])/len(x)
def mne(x):
    return np.mean([n for n in x if n < np.inf])
def feo(x):
    return mne(x)/rel(x)


# In[6]:


NUM_RUNS = 1000
maxeval = 100


# ## 1. Hyperparameter tuning 
# ### Tuning of crossover, correction and mutation hyperparameters for GO 

# Subsequent code uses different **crossovers, corrections and mutations**. The user can choose different settings in either group. 
# 
# * Crossover options are: mix crossover Crossover(), RandomCombination() and UniformMultipoint($k$), where parameter $k \in \mathbb{N}$ is set by user,
# 
# * correction options are: vanilla correction Correction(of), MirrorCorrection(of) and ExtensionCorrection(of),
# 
# * mutations options are: CauchyMutation($\gamma$) and LevyMutation($\gamma$), where parameter $\gamma > 0$ is set by user.
# 
# In the current settings there are 5 crossover, 3 correction and 2 mutation options.

# In[7]:


multipoints = [1, 2, 3]
crossovers = [
    {'crossover': Crossover(), 'name': 'mix'},
    {'crossover': RandomCombination(), 'name': 'rnd'}]
for multipoint in multipoints:
    crossover = {'crossover': UniformMultipoint(multipoint), 'name': 'uni{}'.format(multipoint)}
    crossovers.append(crossover)

corrections = [
    {'correction': Correction(tsp), 'name': 'vanilla'},
    {'correction': MirrorCorrection(tsp), 'name': 'mirror'},
    {'correction': ExtensionCorrection(tsp), 'name': 'extension'}]

parameters = [1, 3, 5]
mutations = []
for parameter in parameters:
    for correction in corrections:
        mutation = {'mutation': CauchyMutation(r=parameter, correction = correction['correction']), 'name': 'cauchy{}_'
                   .format(parameter)+correction['name']}
        mutations.append(mutation)
        mutation = {'mutation': LevyMutation(scale=parameter, correction = correction['correction']), 'name': 'levy{}_'
                   .format(parameter)+correction['name']}
        mutations.append(mutation)


# In[8]:


results = pd.DataFrame()
for crossover in crossovers:
    for mutation in mutations:
        heur_name = 'GO_mut:({})_cro:{}'.format(mutation['name'], crossover['name'])
        runs = []
        for i in tqdm_notebook(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
            run = GeneticOptimization(tsp, maxeval, N=5, M=15, Tsel1=1.0, Tsel2=0.5, 
                                      mutation=mutation['mutation'],
                                      crossover=crossover['crossover']).search()
            run['run'] = i
            run['heur'] = heur_name
            run['mutation'] = mutation['name']
            run['crossover'] = crossover['name']
            runs.append(run)

        res_df = pd.DataFrame(runs, columns=['heur', 'run', 'mutation', 'crossover','best_x', 'best_y', 'neval'])
        results = pd.concat([results, res_df], axis=0)


# In[9]:


results_pivot = results.pivot_table(
    index=['heur', 'mutation', 'crossover'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by='feo')


# The best-performing heuristics are:

# In[10]:


results_pivot.sort_values(by='feo').head(10)


# Goal 1: Implemented **LévyMutation(scale, correction)** does not perform as well as **CauchyMutation(r, correction)**.

# The best-performing and chosen hyperparameter options are: 
# 
# * crossover: UniformMultipoint(1)
# 
# * mutation: CauchyMutation(r=1, correction=Correction())
# 
# * correction: vanilla Correction()

# ### Tuning of $N$, $M$, $T_1$ and $T_2$ hyperparameters for GO with chosen crossover, correction and mutation

# In[11]:


mutation=CauchyMutation(r=1, correction=Correction(tsp))
crossover=UniformMultipoint(1)


# Subsequent code uses different options of **$N$, $M$, $T_1$** and **$T_2$**. The user can choose different settings in either group. 
# 
# In the current settings there are 3 options of $N$, 4 of $M$, 2 of $T_1$ and 2 of $T_2$ options.

# In[12]:


N = [2, 3, 5]
M = [6, 10, 50, 200]
T1 = [5, 0.5]
T2 = [0.4, 0.2]


# In[13]:


results = pd.DataFrame()
for m in M:
    for n in N:
        for temp1 in T1:
            for temp2 in T2:         
                heur_name = 'GO_N:{}_M:{}_T1:{}_T2:{}'.format(n, m, temp1, temp2)
                runs = []
                for i in tqdm_notebook(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
                    run = GeneticOptimization(tsp, maxeval, N=n, M=m, Tsel1=temp1, Tsel2=temp2, 
                                              mutation=mutation,
                                              crossover=crossover).search()
                    run['run'] = i
                    run['heur'] = heur_name
                    run['n'] = n
                    run['m'] = m
                    run['temp1'] = temp1
                    run['temp2'] = temp2
                    runs.append(run)

                res_df = pd.DataFrame(runs, columns=['heur', 'run', 'n', 'm','temp1', 'temp2', 'best_x', 'best_y', 'neval'])
                results = pd.concat([results, res_df], axis=0)


# In[14]:


results_pivot = results.pivot_table(
    index=['heur', 'n', 'm', 'temp1', 'temp2'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by='feo')


# The best-performing and chosen hyperparameter options are: 
# 
# * $N = 3$
# 
# * $M = 10$
# 
# * $T_1 = 0.5$
# 
# * $T_2 = 0.4$

# The best-performing hyperparameters for **GO** on **``TSPGrid(3, 3)``** are:

# In[15]:


results_pivot.sort_values(by=['feo']).head(1)


# Thus, the chosen **GO** is a heuristic with hyperparameters:
# 
# * crossover: UniformMultipoint(2)
# 
# * mutation: CauchyMutation(r=1, correction=Correction())
# 
# * correction: vanilla Correction()
# 
# * $N = 3$
# 
# * $M = 10$
# 
# * $T_1 = 0.5$
# 
# * $T_2 = 0.4$
# 
# This settings of **GO** will be used for comparison between **GO** and **Elitism**.

# In[7]:


mutation=CauchyMutation(r=1, correction=Correction(tsp))
crossover=UniformMultipoint(1)
N=3
M=10
T1=0.5
T2=0.4


# ## 2. Elitism & 3. Parasitism

# In this proposal, **Elitism** is understood as a free passage of a certain percentage of best individuals into the next population without mutation or crossover operations. On the other hand, **Parasitism** means a free passage of the worst individuals.
# 
# For serious comparsion between GO and Elitism we will increase ``NUM_RUNS`` to 5,000.

# In[8]:


from heur_elitism import Elitism, Elitism_Parasitism 


# In[9]:


NUM_RUNS = 5000
maxeval = 100


# In[10]:


results = pd.DataFrame()
runs = []
heur_name = 'GO'
for i in tqdm_notebook(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
    run = GeneticOptimization(tsp, maxeval, N=N, M=M, Tsel1=T1, Tsel2=T2, 
                              mutation=mutation,
                              crossover=crossover).search()
    run['run'] = i
    run['heur'] = heur_name
    run['N'] = N
    run['M'] = M
    run['T1'] = T1
    run['T2'] = T2
    runs.append(run)    
res_df = pd.DataFrame(runs, columns=['heur', 'run', 'N', 'M','T1', 'T2', 'best_x', 'best_y', 'neval'])
results = pd.concat([results, res_df], axis=0)

ratios = [0.3, 0.6, 1] 
#relates to [1, 2, 3] elites
#corresponds to [10 %, 20 %, 30 %] in the new population
for ratio in ratios:
    heur_name = 'ELIT_ratio:{}'.format(ratio)
    for i in tqdm_notebook(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
        run = Elitism(tsp, maxeval, N=N, M=M, ratio=ratio, Tsel1=T1, Tsel2=T2, 
                             mutation=mutation, crossover=crossover).search()
        run['run'] = i
        run['heur'] = heur_name
        run['N'] = N
        run['M'] = M
        run['T1'] = T1
        run['T2'] = T2
        runs.append(run)   
    res_df = pd.DataFrame(runs, columns=['heur', 'run', 'N', 'M','T1', 'T2', 'best_x', 'best_y', 'neval'])
    results = pd.concat([results, res_df], axis=0)

#relates to [1, 2, 3] parasites
#corresponds to [10 %, 20 %, 30 %] in the new population
for ratio in ratios:
    heur_name = 'ELIT_PARAS_ratio:{}'.format(ratio)
    for i in tqdm_notebook(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
        run = Elitism_Parasitism(tsp, maxeval, N=N, M=M, ratio=ratio, Tsel1=T1, Tsel2=T2, 
                             mutation=mutation, crossover=crossover).search()
        run['run'] = i
        run['heur'] = heur_name
        run['N'] = N
        run['M'] = M
        run['T1'] = T1
        run['T2'] = T2
        runs.append(run)   
    res_df = pd.DataFrame(runs, columns=['heur', 'run', 'N', 'M','T1', 'T2', 'best_x', 'best_y', 'neval'])
    results = pd.concat([results, res_df], axis=0)


# In[11]:


results_pivot = results.pivot_table(
    index=['heur', 'N', 'M'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by=['feo'])


# Let $N$ and $M$ be much higher in order to evaluate the goodness of Elitism. The code will be repeated for previous kind of settings.

# In[12]:


N=200
M=1000
T1=0.5
T2=0.2
mutation=CauchyMutation(r=1, correction=Correction(tsp))
crossover=UniformMultipoint(1)


# In[13]:


results = pd.DataFrame()
runs = []
heur_name = 'GO'
for i in tqdm_notebook(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
    run = GeneticOptimization(tsp, maxeval, N=N, M=M, Tsel1=T1, Tsel2=T2, 
                              mutation=mutation,
                              crossover=crossover).search()
    run['run'] = i
    run['heur'] = heur_name
    run['N'] = N
    run['M'] = M
    run['T1'] = T1
    run['T2'] = T2
    runs.append(run)    
res_df = pd.DataFrame(runs, columns=['heur', 'run', 'N', 'M','T1', 'T2', 'best_x', 'best_y', 'neval'])
results = pd.concat([results, res_df], axis=0)

ratios = [0.125, 0.25, 0.5, 0.75, 1] 
#relates to [25, 50, 100, 150, 200] elites
#corresponds to [2.5 %, 5 %, 10 %, 15 %, 20 %] in the new population
for ratio in ratios:
    heur_name = 'ELIT_ratio:{}'.format(ratio)
    for i in tqdm_notebook(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
        run = Elitism(tsp, maxeval, N=N, M=M, ratio=ratio, Tsel1=T1, Tsel2=T2, 
                             mutation=mutation, crossover=crossover).search()
        run['run'] = i
        run['heur'] = heur_name
        run['N'] = N
        run['M'] = M
        run['T1'] = T1
        run['T2'] = T2
        runs.append(run)   
    res_df = pd.DataFrame(runs, columns=['heur', 'run', 'N', 'M','T1', 'T2', 'best_x', 'best_y', 'neval'])
    results = pd.concat([results, res_df], axis=0)

#relates to [25, 50, 100, 150, 200] parasites
#corresponds to [2.5 %, 5 %, 10 %, 15 %, 20 %] in the new population
for ratio in ratios:
    heur_name = 'ELIT_PARAS_ratio:{}'.format(ratio)
    for i in tqdm_notebook(range(NUM_RUNS), 'Testing {}'.format(heur_name)):
        run = Elitism_Parasitism(tsp, maxeval, N=N, M=M, ratio=ratio, Tsel1=T1, Tsel2=T2, 
                             mutation=mutation, crossover=crossover).search()
        run['run'] = i
        run['heur'] = heur_name
        run['N'] = N
        run['M'] = M
        run['T1'] = T1
        run['T2'] = T2
        runs.append(run)   
    res_df = pd.DataFrame(runs, columns=['heur', 'run', 'N', 'M','T1', 'T2', 'best_x', 'best_y', 'neval'])
    results = pd.concat([results, res_df], axis=0)


# In[14]:


results_pivot = results.pivot_table(
    index=['heur', 'N', 'M'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by=['feo'])


# For Elitism, literature suggests that the number of elites in the population should not exceed 10 % of the total population to maintain diversity, which was proven by experiment. **Elitism** performs best for free passage of 25 % of the best individuals, meaning it constitutes 5 % of the new population (second best: 12.5 % and 2.5 % respectively). 
# 
# **Elitism** combined with **Parasitism** performs best for free passage of all 100 % of the best and worst individuals. 
# 
# With correct settings, both modifications outperform 'vanilla' **GO**, although **Elitism** performs overall better.

# ## Conclusions
# * Lévy mutation performs not as good as Cauchy mutation,
# * **Elitism** proposal performs better than 'vanilla' **Genetic Optimization**,
# * **Elitism** in combination with **Parasitism** performs better than 'vanilla' **Genetic Optimization** with correct settings.

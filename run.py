#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pandas as pd
import pickle
import genetic_algorithm as ga


# In[2]:


col_names = ['#','Run', 'Pop Size', 'K', 'Fitness Func', 'Crossover Type', 'Crossover Prob', 'Elitism Ratio',             'Max Parent Allow', 'Mutation Prob', 'Mutation Rate', 'Max Iter', 'Solution Costs', 'Best Solution',             'Stopping Cond', 'Run Time','Min Cost']


# In[3]:


res_df = pd.DataFrame(columns=col_names)
res_df.to_pickle('results.pickle')


# In[4]:


import ray
ray.init()


# ### tuning parameters

# In[5]:


population_size_list = [20,40]

# crossover parameters
crossover_probability_list = [0.5, 0.8]
crossover_type_list = [1, 2] 
fitness_function_list = ['cost', 'distance']

# selection parameters
elitism_ratio_list = [0.2, 0.4] # must be even!
max_parent_allowance_list = [0.125, 0.40]

#mutation parameters
mutation_probability_list = [0.05, 0.1] 
mutation_rate_list = [0.05, 0.2] 

# algorithm settings

max_iter = 500 
k = 10 
individual_size = 120
stop_thrs = 10
mutation_level = 0.1


# In[6]:


params_pool = []
errored_params_pool = []

for population_size in population_size_list:
    for fitness_function in fitness_function_list:
        for crossover_type in crossover_type_list: 
            for crossover_probability in crossover_probability_list:
                for elitism_ratio in elitism_ratio_list:
                    for max_parent_allowance in max_parent_allowance_list:
                        for mutation_probability in mutation_probability_list:
                            for mutation_rate in mutation_rate_list:
                                parameters = [
                                population_size, k,\
                                fitness_function,  crossover_type, crossover_probability, \
                                elitism_ratio, max_parent_allowance, \
                                mutation_probability, mutation_rate, max_iter, \
                                individual_size, stop_thrs, mutation_level \
                                ]

                                params_pool.append(parameters)


# In[7]:


len(params_pool)


# In[8]:


res_df = pd.read_pickle("results.pickle")
len(res_df)


# In[9]:


idx = 0


# In[10]:


@ray.remote
def run(parameters):
    global idx
    idx += 1 
    try:
        res = ga.run(parameters)
    except:
        print('An error occuered with ', parameters)
        errored_params_pool.append(parameters)
        return {}
    print('run',idx, parameters, ' - ',res['run_time'])
    return res


# In[ ]:


start = time.time()
futures = [run.remote(params) for params in params_pool]
results = ray.get(futures)
end =time.time()
print(end-start)


# res_df = pd.DataFrame(columns=col_names)
# res_df.to_pickle('results.pickle')

# ### results

# In[ ]:


run_num = 1

index = len(res_df)

for res, params in zip(results, params_pool):
    
    if len(res)==0:
        solns = [index, run_num] + params[0:10] + ['']*5
    else:    
        solns = [index, run_num] + params[0:10] + [res['cost_func'], res['best_soln'],                                       res['stop_cond'], res['run_time'], res['min_cost']]
    res_df = res_df.append(pd.DataFrame([solns], columns=col_names),ignore_index=True)
    
    index +=1
    
res_df.to_pickle('results.pickle')


# In[ ]:


res_df.head(2)


# #errored_params_pool = []
# open_file = open('errored_params', "wb")
# pickle.dump(errored_params_pool, open_file)
# open_file.close()

# In[ ]:


open_file = open('errored_params', "rb")
errored_params = pickle.load(open_file)
open_file.close()

errored_params = errored_params + errored_params_pool

open_file = open('errored_params', "wb")
pickle.dump(errored_params, open_file)
open_file.close()


# In[ ]:


len(res_df)


# params = [
# population_size, k,\
# fitness_function,  crossover_type, crossover_probability, \
# elitism_ratio, max_parent_allowance, \
# mutation_probability, mutation_rate, max_iter, \
# individual_size, stop_thrs, mutation_level \
# ]

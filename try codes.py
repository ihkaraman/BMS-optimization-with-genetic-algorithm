#!/usr/bin/env python
# coding: utf-8

# In[1]:


import genetic_algorithm as ga


# In[2]:


population_size = 20

# crossover parameters
crossover_probability = 0.6
crossover_type = 1
fitness_function = 'cost'

# selection parameters
elitism_ratio = 0.2 # must be even!
max_parent_allowance = 0.125

#mutation parameters
mutation_probability = 0.1 
mutation_rate = 0.1

# algorithm settings
max_iter = 300 
k = 10 
stop_thrs = 10
mutation_level = 0.2

run_num = 3
run = 1

individual_size = 120


# In[3]:


parameters = [
run, population_size, k,\
fitness_function,  crossover_type, crossover_probability, \
elitism_ratio, max_parent_allowance, \
mutation_probability, mutation_rate, max_iter, individual_size,\
stop_thrs, mutation_level \
]


# In[4]:


ga.run(parameters)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Import Libraries
import cplex
import math
import numpy as np
import pandas as pd
import random


# In[9]:


# Parameter sets are partitioned in this script.
# Optional Settings / Parameters
class OptionalParameters(object):
    def __init__(self):
        # Time
        self.time_unit = 'H'  # 'H' for hourly periods, 'M' for 15 min periods
        self.time_horizon = 120  # unit: hours

        # Generation
        self.option_diesel = 1
        self.option_solar = 1

        # Trade
        self.option_purchase = 1
        self.option_sell = 1

        # Degradation
        # Complex depredation function will be included in future versions
        self.degradation_unit = 1
        self.degradation_complex = 0


# In[10]:


# Real Time Parameters
# Including Forecasts
class RealTimeParameters(object):
    def __init__(self):
        # To test (get from arrays)
        # 'T' - for test
        self.test = 'T'

        # Real time parameters are constructed considering the time intervals and time horizon
        # Time
        self.time_unit = 'H'  # 'H' for hourly periods, 'M' for 15 min periods
        self.time_horizon = 10  # unit: hours (no use yet)

        # Consumption
        # Power to kwh conversion is needed !
        # Mean and Variance are used to identify uncertainty.
        if self.test == 'T':
            self.consumption = pd.read_csv("miris_load_hour.csv")
            self.consumption = list(self.consumption['Rounded'])

        # Committed Generation (Conventional and Renewable Resources can be identified in detail.)
        # Power to kwh conversion is needed !
        # Mean and Variance are used to identify uncertainty.
        # Uncertainty of conventional resources are ignored.
        if self.test == 'T':
            self.generation_solar = pd.read_csv("miris_PV_Hour.csv")
            self.generation_solar = list(self.generation_solar['Rounded'])

        # Price
        # 1D Array len:time horizon
        self.price_purchase = 1000

        self.price_sell = 1000

        self.price_diesel = 1000

        # Battery
        if self.test == 'T':
            # Initial load is set to 0 - Battery has no load to discharge
            self.initial_load = 0


# In[11]:


# Technical Parameters
# These parameters are not case depended.
# eg. battery characteristics
class ModelParameters(object):
    def __init__(self):
        # Battery Technical Characteristics
        self.battery_capacity_max = 130000
        self.battery_capacity_min = 0
        self.round_trip_efficiency = 0.9
        self.battery_replacement_cost = 100
        self.battery_lifetime_throughput = 647400

        # Unit battery cost
        self.unit_battery_cost = 1000

        # Grid
        self.transmission_capacity = 1000

        # Uncertainty
        # Precision of modeling uncertainty can be relaxed with changing z-value
        self.z_value = 2.58  # for probability 0.99


# In[13]:


# Objective function value of the determined l-sequence. 
def calculate_fitness(l_array):
    
    l_array = l_array.astype('float64')
    
    model = cplex.Cplex()
    options_p = OptionalParameters()
    real_p = RealTimeParameters()
    model_p = ModelParameters()

    ##########

    # Create decision variables
    dvar = model.variables.type

    # Size of the battery
    model.variables.add(names=["b"])
    model.variables.set_lower_bounds("b", 0.0)
    model.variables.set_types("b", dvar.continuous)

    # Charge/discharge decision in period t
    # 1 - charge, 0 - discharge
    model.variables.add(names=["y" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("y" + '_' + str(t), 0.0)
        model.variables.set_types("y" + '_' + str(t), dvar.binary)

    # Amount of charge in period t
    model.variables.add(names=["yc" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("yc" + '_' + str(t), 0.0)
        model.variables.set_types("yc" + '_' + str(t), dvar.continuous)

    # Amount of discharge in period t
    model.variables.add(names=["yd" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("yd" + '_' + str(t), 0.0)
        model.variables.set_types("yd" + '_' + str(t), dvar.continuous)

    # Amount of generation of dispatchable/diesel generator in period t
    model.variables.add(names=["gd" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("gd" + '_' + str(t), 0.0)
        model.variables.set_types("gd" + '_' + str(t), dvar.continuous)

    # Purchase/sell decision in period t
    # 1 - purchase, 0 - sell
    model.variables.add(names=["x" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("x" + '_' + str(t), 0.0)
        model.variables.set_types("x" + '_' + str(t), dvar.binary)

    # Amount of energy purchased from the grid (interconnected)
    model.variables.add(names=["xp" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("xp" + '_' + str(t), 0.0)
        model.variables.set_types("xp" + '_' + str(t), dvar.continuous)

    # Amount of energy sold to the gird (interconnected)
    model.variables.add(names=["xs" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("xs" + '_' + str(t), 0.0)
        model.variables.set_types("xs" + '_' + str(t), dvar.continuous)

    # State of charge of the battery at the end of period t
    model.variables.add(names=["l" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("l" + '_' + str(t), 0.0)
        model.variables.set_types("l" + '_' + str(t), dvar.continuous)

    ##########

    # OBJECTIVE FUNCTION

    # Add objective function and set its sense
    model.objective.set_sense(model.objective.sense.minimize)

    # Battery Fixed Cost (per kW)
    model.objective.set_linear([("b", model_p.battery_replacement_cost)])

    # Trade #
    if options_p.option_purchase == 1:
        for t in range(options_p.time_horizon):
            model.objective.set_linear([("xp" + '_' + str(t), real_p.price_purchase)])

    if options_p.option_sell == 1:
        for t in range(options_p.time_horizon):
            model.objective.set_linear([("xs" + '_' + str(t), real_p.price_sell)])

    # Generation #
    # Diesel
    if options_p.option_diesel == 1:
        for t in range(options_p.time_horizon):
            model.objective.set_linear([("gd" + '_' + str(t), real_p.price_diesel)])

    # Degradation #
    model.objective.set_linear([("b", model_p.unit_battery_cost)])
    b_cost = model_p.battery_replacement_cost / (model_p.battery_lifetime_throughput *
                                                 math.sqrt(model_p.round_trip_efficiency))
    if options_p.degradation_unit == 1:
        for t in range(options_p.time_horizon):
            model.objective.set_linear([("yc" + '_' + str(t), b_cost)])

    ##########

    # CONSTRAINTS

    # Constraint 1: Supply - Demand Balance
    # Constraint set involves optional elements
    for t in range(options_p.time_horizon):
        C1_lin = []
        C1_lin.append("yd" + '_' + str(t))
        C1_lin.append("xp" + '_' + str(t))
        C1_lin.append("yc" + '_' + str(t))
        C1_lin.append("xs" + '_' + str(t))
        if options_p.option_diesel == 1:
            C1_lin.append("gd" + '_' + str(t))
        C1_val = []
        C1_val.append(1)
        C1_val.append(1)
        C1_val.append(-1)
        C1_val.append(-1)
        if options_p.option_diesel == 1:
            C1_val.append(1)
        C1_rhs = real_p.consumption[t]
        # Solar Power
        if options_p.option_solar == 1:
            C1_rhs += -1 * real_p.generation_solar[t]

        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind=C1_lin, val=C1_val)],
                                     rhs=[C1_rhs],
                                     names=["C1"], senses=['E'])

    # Constraint 2: Battery Characteristics
    # Charge or Discharge
    for t in range(options_p.time_horizon):
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["yc" + '_' + str(t), "y" + '_' + str(t)],
                                       val=[1, -1 * (model_p.battery_capacity_max - model_p.battery_capacity_min)])],
            rhs=[0], names=["C2_1"], senses=['L'])

    for t in range(options_p.time_horizon):
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["yd" + '_' + str(t), "y" + '_' + str(t)],
                                       val=[1, (model_p.battery_capacity_max - model_p.battery_capacity_min)])],
            rhs=[(model_p.battery_capacity_max - model_p.battery_capacity_min)], names=["C2_2"], senses=['L'])

    # Constraint 3: Load Balance
    # First constraint involves the initial load parameter therefore it is separated.

    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["l_0", "yc_0", "yd_0"],
                                                            val=[1, -1, 1])],
                                 rhs=[real_p.initial_load], names=["C3_1"], senses=['E'])
    for t in range(1, options_p.time_horizon):
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["l" + '_' + str(t), "l" + '_' + str(t - 1),
                                                                     "yc" + '_' + str(t), "yd" + '_' + str(t)],
                                                                val=[1, -1, -1, 1])], rhs=[0],
                                     names=["C3_2"], senses=['E'])

    # Constraint 4: Battery Characteristics (Maximum and Minimum State of Charge/Capacity)
    for t in range(options_p.time_horizon):
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["l" + '_' + str(t)], val=[1])],
                                     rhs=[model_p.battery_capacity_min], names=["C4_1"], senses=['G'])
    for t in range(options_p.time_horizon):
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["l" + '_' + str(t), "b"], val=[1, -1])],
                                     rhs=[0], names=["C4_2"], senses=['L'])
    '''
    # Decide on Battery Capacity
    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["b"], val=[1])],
                                 rhs=[initial_size], names=["C_B"], senses=['E'])
    '''
    # Decide on Battery Load  
    for t in range(options_p.time_horizon):
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["l" + '_' + str(t)], val=[1])],
                                     rhs=[l_array[t]], names=["C_B"], senses=['E'])
    
    '''
    # Constraint 5: Grid Characteristics
    # Purchase or Sell
    for t in range(options_p.time_horizon):
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["xp" + '_' + str(t), "y" + '_' + str(t)],
                                                                val=[1, -1 * model_p.transmission_capacity])],
                                     rhs=[0], names=["C5_1"], senses=['L'])

        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["xs" + '_' + str(t), "yd" + '_' + str(t)],
                                                                val=[1, -1])],
                                     rhs=[0], names=["C5_2"], senses=['L'])
    '''

    # Export model in .lp format
    # To check the code manually
    #model.write('Model.lp')
    
    out = model.set_log_stream(None)
    out = model.set_error_stream(None)
    out = model.set_warning_stream(None)
    out = model.set_results_stream(None)
    
    # Solve Model
    
    model.solve()

    solution = model.solution.get_objective_value()/1000000    
    
    return solution


# In[14]:


# Stochastic Mixed Integer Programming Model
def calculate_objective_function(initial_size):

    model = cplex.Cplex()
    options_p = OptionalParameters()
    real_p = RealTimeParameters()
    model_p = ModelParameters()

    ##########

    # Create decision variables
    dvar = model.variables.type

    # Size of the battery
    model.variables.add(names=["b"])
    model.variables.set_lower_bounds("b", 0.0)
    model.variables.set_types("b", dvar.continuous)

    # Charge/discharge decision in period t
    # 1 - charge, 0 - discharge
    model.variables.add(names=["y" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("y" + '_' + str(t), 0.0)
        model.variables.set_types("y" + '_' + str(t), dvar.binary)

    # Amount of charge in period t
    model.variables.add(names=["yc" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("yc" + '_' + str(t), 0.0)
        model.variables.set_types("yc" + '_' + str(t), dvar.continuous)

    # Amount of discharge in period t
    model.variables.add(names=["yd" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("yd" + '_' + str(t), 0.0)
        model.variables.set_types("yd" + '_' + str(t), dvar.continuous)

    # Amount of generation of dispatchable/diesel generator in period t
    model.variables.add(names=["gd" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("gd" + '_' + str(t), 0.0)
        model.variables.set_types("gd" + '_' + str(t), dvar.continuous)

    # Purchase/sell decision in period t
    # 1 - purchase, 0 - sell
    model.variables.add(names=["x" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("x" + '_' + str(t), 0.0)
        model.variables.set_types("x" + '_' + str(t), dvar.binary)

    # Amount of energy purchased from the grid (interconnected)
    model.variables.add(names=["xp" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("xp" + '_' + str(t), 0.0)
        model.variables.set_types("xp" + '_' + str(t), dvar.continuous)

    # Amount of energy sold to the gird (interconnected)
    model.variables.add(names=["xs" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("xs" + '_' + str(t), 0.0)
        model.variables.set_types("xs" + '_' + str(t), dvar.continuous)

    # State of charge of the battery at the end of period t
    model.variables.add(names=["l" + '_' + str(t) for t in range(options_p.time_horizon)])
    for t in range(options_p.time_horizon):
        model.variables.set_lower_bounds("l" + '_' + str(t), 0.0)
        model.variables.set_types("l" + '_' + str(t), dvar.continuous)

    ##########

    # OBJECTIVE FUNCTION

    # Add objective function and set its sense
    model.objective.set_sense(model.objective.sense.minimize)

    # Battery Fixed Cost (per kW)
    model.objective.set_linear([("b", model_p.battery_replacement_cost)])

    # Trade #
    if options_p.option_purchase == 1:
        for t in range(options_p.time_horizon):
            model.objective.set_linear([("xp" + '_' + str(t), real_p.price_purchase)])

    if options_p.option_sell == 1:
        for t in range(options_p.time_horizon):
            model.objective.set_linear([("xs" + '_' + str(t), real_p.price_sell)])

    # Generation #
    # Diesel
    if options_p.option_diesel == 1:
        for t in range(options_p.time_horizon):
            model.objective.set_linear([("gd" + '_' + str(t), real_p.price_diesel)])

    # Degradation #
    model.objective.set_linear([("b", model_p.unit_battery_cost)])
    b_cost = model_p.battery_replacement_cost / (model_p.battery_lifetime_throughput *
                                                 math.sqrt(model_p.round_trip_efficiency))
    if options_p.degradation_unit == 1:
        for t in range(options_p.time_horizon):
            model.objective.set_linear([("yc" + '_' + str(t), b_cost)])

    ##########

    # CONSTRAINTS

    # Constraint 1: Supply - Demand Balance
    # Constraint set involves optional elements
    for t in range(options_p.time_horizon):
        C1_lin = []
        C1_lin.append("yd" + '_' + str(t))
        C1_lin.append("xp" + '_' + str(t))
        C1_lin.append("yc" + '_' + str(t))
        C1_lin.append("xs" + '_' + str(t))
        if options_p.option_diesel == 1:
            C1_lin.append("gd" + '_' + str(t))
        C1_val = []
        C1_val.append(1)
        C1_val.append(1)
        C1_val.append(-1)
        C1_val.append(-1)
        if options_p.option_diesel == 1:
            C1_val.append(1)
        C1_rhs = real_p.consumption[t]
        # Solar Power
        if options_p.option_solar == 1:
            C1_rhs += -1 * real_p.generation_solar[t]

        model.linear_constraints.add(lin_expr = [cplex.SparsePair(ind=C1_lin, val=C1_val)],
                                     rhs=[C1_rhs],
                                     names=["C1"], senses=['E'])

    # Constraint 2: Battery Characteristics
    # Charge or Discharge
    for t in range(options_p.time_horizon):
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["yc" + '_' + str(t), "y" + '_' + str(t)],
                                       val=[1, -1 * (model_p.battery_capacity_max - model_p.battery_capacity_min)])],
            rhs=[0], names=["C2_1"], senses=['L'])

    for t in range(options_p.time_horizon):
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["yd" + '_' + str(t), "y" + '_' + str(t)],
                                       val=[1, (model_p.battery_capacity_max - model_p.battery_capacity_min)])],
            rhs=[(model_p.battery_capacity_max - model_p.battery_capacity_min)], names=["C2_2"], senses=['L'])

    # Constraint 3: Load Balance
    # First constraint involves the initial load parameter therefore it is separated.

    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["l_0", "yc_0", "yd_0"],
                                                            val=[1, -1, 1])],
                                 rhs=[real_p.initial_load], names=["C3_1"], senses=['E'])
    for t in range(1, options_p.time_horizon):
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["l" + '_' + str(t), "l" + '_' + str(t - 1),
                                                                     "yc" + '_' + str(t), "yd" + '_' + str(t)],
                                                                val=[1, -1, -1, 1])], rhs=[0],
                                     names=["C3_2"], senses=['E'])

    # Constraint 4: Battery Characteristics (Maximum and Minimum State of Charge/Capacity)
    for t in range(options_p.time_horizon):
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["l" + '_' + str(t)], val=[1])],
                                     rhs=[model_p.battery_capacity_min], names=["C4_1"], senses=['G'])
    for t in range(options_p.time_horizon):
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["l" + '_' + str(t), "b"], val=[1, -1])],
                                     rhs=[0], names=["C4_2"], senses=['L'])

    # Decide on Battery Capacity
    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["b"], val=[1])],
                                 rhs=[initial_size], names=["C_B"], senses=['E'])

    '''
    # Constraint 5: Grid Characteristics
    # Purchase or Sell
    for t in range(options_p.time_horizon):
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["xp" + '_' + str(t), "y" + '_' + str(t)],
                                                                val=[1, -1 * model_p.transmission_capacity])],
                                     rhs=[0], names=["C5_1"], senses=['L'])

        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["xs" + '_' + str(t), "yd" + '_' + str(t)],
                                                                val=[1, -1])],
                                     rhs=[0], names=["C5_2"], senses=['L'])
    '''

    # Export model in .lp format
    # To check the code manually
    #model.write('Model.lp')
    
    out = model.set_log_stream(None)
    out = model.set_error_stream(None)
    out = model.set_warning_stream(None)
    out = model.set_results_stream(None)
    # Solve Model
    model.solve()
    l_values = []

    for t in range(options_p.time_horizon):
        l_values.append(model.solution.get_values(1, "l"+'_' + str(t)))
    solution = model.solution.get_objective_value()
    '''
    # Create Solution Pool
    numsol = model.solution.pool.get_num()
    model.populate_solution_pool()
    print("The solution pool contains %d solutions." % numsol)
    # mean_obj_value = model.solution.pool.get_mean_objective_value()
    # print(mean_obj_value)

    # Get Pool Values - only B and L

    solution_pool = []
    for i in range(numsol):
        soln_list = []
        # First element is the objective function value
        soln_list.append(model.solution.pool.get_objective_value(i))
        # Second Element is the battery size
        soln_list.append(model.solution.pool.get_values(i, "b"))
        # Rest of the elements are L values (battery load at time t)
        for t in range(options_p.time_horizon):
            soln_list.append(model.solution.pool.get_values(i, "l" + '_' + str(t)))
        if i == 0:
            solution_pool = np.array(soln_list)
        else:
            solution_pool = np.vstack((solution_pool, soln_list))
    print(solution_pool)
    '''
    
    return solution/1000000


# In[34]:


def generate_random_array(pop_size):
    
    max_consumption = max(RealTimeParameters().consumption)
    max_generation_solar = max(RealTimeParameters().generation_solar)
    
    l_upper_limit = max(max_consumption, max_generation_solar)
    l_lower_limit = 0 
    
    random_array = np.random.randint(l_lower_limit, l_upper_limit, size = (pop_size, OptionalParameters().time_horizon))
    
    return random_array


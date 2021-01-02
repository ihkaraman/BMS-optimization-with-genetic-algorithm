# RATIO TECHNOLOGIES - BATTERY SCHEDULING OPTIMIZATION
# Version 2

# Import Libraries
import pandas as pd
import numpy as np
import math
from time import time
from ortools.linear_solver import pywraplp


# Parameter sets are partitioned in this script.
# Optional Settings / Parameters
class OptionalParameters(object):
    def __init__(self):
        # Time
        self.time_unit = 'M'  # 'H' for hourly periods, 'M' for 15 min periods
        self.time_horizon = 6  # unit: hours
        # Conversion of hourly time horizon to 15 min
        if self.time_unit == 'M':
            self.time_horizon = self.time_horizon * 4
        # Generation
        self.option_diesel = 1
        self.option_solar = 1
        self.option_wind = 0
        self.option_conventional = 0

        # Trade
        self.option_purchase = 0
        self.option_sell = 1

        # Uncertainty
        self.option_uncertainty = 1
        self.option_wind_uncertainty = 0
        self.option_solar_uncertainty = 0
        self.option_consumption_uncertainty = 1

        # Degradation
        # Complex depredation function will be included in future versions
        self.degradation_unit = 1
        self.degradation_complex = 0


# Real Time Parameters
# Including Forecasts
class RealTimeParameters(object):
    def __init__(self, load_forecast, pv_forecast, price, soc):
        # To test (get from arrays)
        # 'T' - for test
        self.test = '0'

        # Real time parameters are constructed considering the time intervals and time horizon
        # Time
        self.time_unit = 'M'  # 'H' for hourly periods, 'M' for 15 min periods
        self.time_horizon = 10  # unit: hours (no use yet)

        # Consumption
        # Power to kwh conversion is needed !
        # Mean and Variance are used to identify uncertainty.
        if self.test == 'T':
            self.consumption = [161.9947708, 164.3771979, 167.6816458, 168.9883438, 171.062625, 170.3892917, 169.58691670000002, 164.92975, 161.2016146, 156.39544790000002, 152.5307188, 153.1185938, 155.0085312, 156.97095829999998, 158.2960625, 160.67164580000002, 162.0785625, 163.8793229, 161.29348960000002, 159.31069789999998, 157.8663229, 156.8883542, 152.34863539999998, 149.2184896, 149.1715625, 147.1031667, 144.9052604, 141.11911460000002, 136.2078125, 133.2229479, 130.5301354, 129.84665619999998, 126.6729583, 122.4273333, 119.6230521, 116.0982188, 111.2675938, 106.8804688, 105.7983229, 101.2806667, 95.9375104, 89.0575938, 82.8199427, 79.3629167, 73.8793438, 66.77714060000001, 63.550151, 64.9705469, 62.3326094, 61.640041700000005, 59.260843799999996, 62.984046899999996, 66.08008849999999, 67.59629170000001, 64.4890104, 61.6406146, 61.0317031, 61.3122865, 59.6516458, 57.0823646, 55.588869800000005, 56.8618906, 58.5539115, 60.706270800000006, 61.3450833, 60.1713125, 57.642093800000005, 55.094625, 52.132963499999995, 49.9022031, 50.7886771, 50.9384948, 49.880474, 50.716187500000004, 51.203099, 51.896593800000005, 52.4329688, 55.475479199999995, 58.1575156, 63.6937344, 66.8315417, 69.04239580000001, 74.0027135, 81.9252083, 87.8340625, 92.6759479, 99.104625, 104.1344062, 108.5217188, 113.6786354, 120.8857708, 125.0309583, 129.8427083, 133.1934583, 133.2766354, 135.40775]
        else:
            self.consumption = load_forecast
        if self.time_unit == 'M':
            for i in range(len(self.consumption)):
                self.consumption[i] = self.consumption[i] / 4
        # Mean and Variance of Consumption Fluctuation
        self.consumption_mean = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        self.consumption_var = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Committed Generation (Conventional and Renewable Resources can be identified in detail.)
        # Power to kwh conversion is needed !
        # Mean and Variance are used to identify uncertainty.
        # Uncertainty of conventional resources are ignored.
        if self.test == 'T':
            self.generation_solar = [165.8503929, 258.39339290000004, 251.302125, 184.6930179, 132.3195, 138.682625, 233.9702857, 267.2459464, 387.5523571, 455.0665357, 491.05289289999996, 467.1266071, 462.6675714, 468.842, 523.9428571, 501.6573929, 426.6855714, 326.70192860000003, 279.8264821, 279.84894640000005, 127.3085982, 89.53684820000001, 158.5214643, 66.8693705, 46.823317, 28.3161473, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.8560625, 68.9671384, 0.0, 9.7257656]
        else:
            self.generation_solar = pv_forecast
        if self.time_unit == 'M':
            for i in range(len(self.generation_solar)):
                self.generation_solar[i] = self.generation_solar[i] / 4
        # Mean and Variance of Solar Fluctuation
        self.generation_solar_mean = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        self.generation_solar_var = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.generation_wind = [0]
        if self.time_unit == 'M':
            for i in range(len(self.generation_wind)):
                self.generation_wind[i] = self.generation_wind[i] / 4
        # Mean and Variance of Wind Fluctuation
        self.generation_wind_mean = 0
        self.generation_wind_var = 0
        self.generation_conventional = 0

        # Price
        # 1D Array len:time horizon
        self.price_purchase = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.price_sell = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.price_diesel = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

        # Battery
        if self.test == 'T':
            self.initial_load = 1300 * 0.90
        else:
            self.initial_load = soc * 0.9


# Technical Parameters
# These parameters are not case depended.
# eg. battery characteristics
class ModelParameters(object):
    def __init__(self):
        # Battery Technical Characteristics
        self.battery_capacity_max = 1300
        self.battery_capacity_min = 0
        self.round_trip_efficiency = 0.9
        self.battery_replacement_cost = 100
        self.battery_lifetime_throughput = 647400

        # Battery Charge/Discharge Speed
        self.battery_speed_level = 3
        self.battery_capacity_speed = [0, 1000, 1300]
        # Generator price may depend on a real time data set, a reconstruction may be needed !
        self.price_battery = [0.1, 0.4, 1]

        # Dispatchable Generator Charge/Discharge Speed
        self.dg_speed_level = 3
        self.dg_capacity_speed = [0, 1000, 1300]
        # Generator price may depend on a real time data set, a reconstruction may be needed !
        self.price_diesel = [0.1, 0.4, 1]

        # Grid
        self.transmission_capacity = 1000

        # Uncertainty
        # Precision of modeling uncertainty can be relaxed with changing z-value
        self.z_value = 2.58  # for probability 0.99


# Stochastic Mixed Integer Programming Model
def ScheduleBattery(load_forecast, pv_forecast, price, soc):
    # Import Parameters
    options_p = OptionalParameters()
    real_p = RealTimeParameters(load_forecast, pv_forecast, price, soc)
    model_p = ModelParameters()

    # print('Controller', real_p.load_test)

    # Starting Time
    t0 = time()

    # CREATE MODEL
    solver = pywraplp.Solver('BatteryScheduling', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    ####################

    # DECISION VARIABLES
    # Charge/discharge decision in period t
    # 1 - charge, 0 - discharge
    y = {}
    for t in range(options_p.time_horizon):
        y[t] = solver.BoolVar('y[%i]' % t)

    # Amount of charge in period t
    yc = {}
    for t in range(options_p.time_horizon):
        yc[t] = solver.NumVar(0, solver.infinity(), 'yc[%i]' % t)

    # Amount of discharge in period t
    yd = {}
    for t in range(options_p.time_horizon):
        yd[t] = solver.NumVar(0, solver.infinity(), 'yd[%i]' % t)

    # Amount of generation of dispatchable/diesel generator in period t
    gd = {}
    for t in range(options_p.time_horizon):
        gd[t] = solver.NumVar(0, solver.infinity(), 'gd[%i]' % t)

    # Amount of energy purchased from the grid (interconnected)
    xp = {}
    for t in range(options_p.time_horizon):
        xp[t] = solver.NumVar(0, solver.infinity(), 'xp[%i]' % t)

    # Amount of energy sold to the gird (interconnected)
    xs = {}
    for t in range(options_p.time_horizon):
        xs[t] = solver.NumVar(0, solver.infinity(), 'xs[%i]' % t)

    # Purchase/sell decision in period t
    # 1 - purchase, 0 - sell
    x = {}
    for t in range(options_p.time_horizon):
        x[t] = solver.BoolVar('x[%i]' % t)

    # State of charge of the battery at the end of period t
    l = {}
    for t in range(options_p.time_horizon):
        l[t] = solver.NumVar(0, solver.infinity(), 'l[%i]' % t)

    # State of charge level of the battery that needs to be conserved in period t in order to handle uncertainty
    lf = {}
    for t in range(options_p.time_horizon):
        lf[t] = solver.NumVar(0, solver.infinity(), 'lf[%i]' % t)

    # Amount of charge with speed level i in period t
    # An auxiliary decision variable to prevent nonlinear objective function
    s = {}
    for t in range(options_p.time_horizon):
        for i in range(model_p.battery_speed_level):
            s[i, t] = solver.NumVar(0, solver.infinity(), 'lf[%i, %i]' % (i, t))

    # Battery speed in period t
    # 1 - speed level i is used, 0 - o/w
    sl = {}
    for t in range(options_p.time_horizon):
        for i in range(model_p.battery_speed_level):
            sl[i, t] = solver.BoolVar('sl[%i, %i]' % (i, t))

    # Amount of generation with dispatchable generator with speed level j in period t
    # An auxiliary decision variable to prevent nonlinear objective function
    gdl = {}
    for t in range(options_p.time_horizon):
        for j in range(model_p.dg_speed_level):
            gdl[j, t] = solver.NumVar(0, solver.infinity(), 'gdl[%i, %i]' % (j, t))

    # Battery speed in period t
    # 1 - speed level i is used, 0 - o/w
    gdb = {}
    for t in range(options_p.time_horizon):
        for j in range(model_p.dg_speed_level):
            gdb[j, t] = solver.BoolVar('gdb[%i, %i]' % (j, t))

    ####################

    # OBJECTIVE FUNCTION
    # Minimize - Total Cost
    # Total Cost = Trade Cost + Generation Cost
    total_cost = solver.Objective()
    total_cost.SetMinimization()

    # Trade #
    if options_p.option_purchase == 1:
        for t in range(options_p.time_horizon):
            total_cost.SetCoefficient(xp[t], real_p.price_purchase[t])
    if options_p.option_sell == 1:
        for t in range(options_p.time_horizon):
            total_cost.SetCoefficient(xs[t], -1 * real_p.price_sell[t])

    # Generation #
    # Diesel
    if options_p.option_diesel == 1:
        for t in range(options_p.time_horizon):
            total_cost.SetCoefficient(gd[t], model_p.price_diesel[j])
    # Battery Speed #
    for j in range(model_p.battery_speed_level):
        for t in range(options_p.time_horizon):
            total_cost.SetCoefficient(s[j, t], model_p.price_battery[j])

    # Degradation #
    b = model_p.battery_replacement_cost / \
        (model_p.battery_lifetime_throughput * math.sqrt(model_p.round_trip_efficiency))
    if options_p.degradation_unit == 1:
        for t in range(options_p.time_horizon):
            total_cost.SetCoefficient(yc[t], b)

    ####################

    # CONSTRAINTS

    # Constraint 1: Supply - Demand Balance
    # Constraint set involves optional elements
    for t in range(options_p.time_horizon):
        # Supply #
        # Discharge
        supply_terms = [yd[t]]
        # Dispatchable/Diesel Generation
        if options_p.option_diesel == 1:
            supply_terms.append(gd[t])
        # Solar Power
        if options_p.option_solar == 1:
            supply_terms.append(real_p.generation_solar[t])
        # Wind Power
        if options_p.option_wind == 1:
            supply_terms.append(real_p.generation_wind[t])
        # Conventional Power Resources
        if options_p.option_conventional == 1:
            supply_terms.append(real_p.generation_conventional[t])
        # Trade - Purchase
        if options_p.option_purchase == 1:
            supply_terms.append(xp[t])
        # Demand #
        # Charge & Consumption
        demand_terms = [yc[t], real_p.consumption[t]]
        # Trade - Sell
        if options_p.option_sell == 1:
            demand_terms.append(xs[t])
        # Constraint
        solver.Add(solver.Sum(supply_terms) == solver.Sum(demand_terms))

    # Constraint 2: Battery Characteristics
    # Charge or Discharge
    for t in range(options_p.time_horizon):
        solver.Add(yc[t] <= (model_p.battery_capacity_max - model_p.battery_capacity_min) * y[t])
        solver.Add(yd[t] <= (model_p.battery_capacity_max - model_p.battery_capacity_min) * (1 - y[t]))

    # Constraint 3: Load Balance
    # First constraint involves the initial load parameter therefore it is separated.
    solver.Add(l[0] == real_p.initial_load + yc[0] - yd[0])
    for t in range(1, options_p.time_horizon):
        solver.Add(l[t] == l[t - 1] + yc[t] - yd[t])

    # Constraint 4: Battery Characteristics (Maximum and Minimum State of Charge/Capacity)
    for t in range(options_p.time_horizon):
        solver.Add(l[t] >= model_p.battery_capacity_min)
        solver.Add(l[t] <= model_p.battery_capacity_max)

    # Constraint 5: Grid Characteristics
    # Purchase or Sell
    for t in range(options_p.time_horizon):
        solver.Add(xp[t] <= x[t] * model_p.transmission_capacity)
        solver.Add(xs[t] <= (1 - x[t]) * model_p.transmission_capacity)

    # Constraint 6: Grid Characteristics
    # Limit selling energy when battery is discharging
    for t in range(options_p.time_horizon):
        solver.Add(xs[t] <= (1 - y[t]) * model_p.transmission_capacity)
        solver.Add(xs[t] <= yd[t])

    # Constraint 7: Battery Speed/Level
    for t in range(options_p.time_horizon):
        solver.Add(yc[t] <= solver.Sum((s[i, t]) for i in range(model_p.battery_speed_level)))
    for t in range(options_p.time_horizon):
        for i in range(model_p.battery_speed_level):
            solver.Add(s[i, t] <= model_p.battery_capacity_speed[i] * sl[i, t])
    # Only one speed level can be assigned in a single period
    # This constraint can also ve modelled with equality
    for t in range(options_p.time_horizon):
        solver.Add(solver.Sum((sl[i, t]) for i in range(model_p.battery_speed_level)) <= 1)

    # Constraint 8: Diesel Generator
    # Speed/Level & State Dependency
    for t in range(options_p.time_horizon):
        solver.Add(gd[t] == solver.Sum((gdl[i, t]) for i in range(model_p.battery_speed_level)))
    for t in range(options_p.time_horizon):
        for j in range(model_p.dg_speed_level):
            solver.Add(gdl[j, t] <= model_p.dg_capacity_speed[j] * gdb[j, t])
    # Only one speed level can be assigned in a single period
    # This constraint can also ve modelled with equality
    for t in range(options_p.time_horizon):
        solver.Add(solver.Sum((gdb[j, t]) for j in range(model_p.dg_speed_level)) <= 1)

    # Constraint 9: Uncertainty
    if options_p.option_uncertainty == 1:
        # Calculate Fluctuation Mean and Variance
        fluctuation_mean = 0
        fluctuation_variance = 0
        for t in range(options_p.time_horizon):
            if options_p.option_solar_uncertainty == 1:
                fluctuation_mean = fluctuation_variance + real_p.generation_solar_mean[t]
                fluctuation_variance = fluctuation_variance + real_p.generation_solar_var[t]
            if options_p.option_wind_uncertainty == 1:
                fluctuation_mean = fluctuation_variance + real_p.generation_wind_mean[t]
                fluctuation_variance = fluctuation_variance + real_p.generation_wind_var[t]
            if options_p.option_consumption_uncertainty == 1:
                fluctuation_mean = fluctuation_variance + real_p.consumption_mean[t]
                fluctuation_variance = fluctuation_variance + real_p.consumption_var[t]
            # Constraint
            solver.Add(lf[t] >= fluctuation_mean + model_p.z_value * fluctuation_variance)
        # State of charge level must be restricted from behind (lower bound)
        for t in range(1, options_p.time_horizon):
            solver.Add(l[t - 1] >= lf[t])

    # Export Model (Validation Purpose) - Comment it to close this property
    # NOT SUPPORTED ANYMORE !!!
    # solver.ExportModelAsLpFormat(False)
    # solver.ExportModelAsMpsFormat(True, False)

    # Solve Model
    solution = solver.Solve()

    # Finishing Time
    t1 = time()
    print('Solution Time', t1-t0)

    # Check Optimality
    if solution != pywraplp.Solver.OPTIMAL:
        print("The problem does not have an optimal solution!")
        exit(1)
    else:
        print('Optimal Objective Function Value = %d' % solver.Objective().Value())

    # Print Solution Set
    # Into Numpy Array
    '''
    # Battery #
    battery_solution_set = ['Period', 'SoC', 'Charge', 'Discharge', 'Reserve (for Uncertainty)']
    for t in range(options_p.time_horizon):
        b_solution_array = [t, l[t].solution_value(), yc[t].solution_value(), yd[t].solution_value(),
                            lf[t].solution_value()]
        battery_solution_set = np.vstack((battery_solution_set, b_solution_array))
    print(battery_solution_set)
    # Trade and Generation
    other_solution_set = ['Period', 'Purchase', 'Sell', 'Diesel']
    for t in range(options_p.time_horizon):
        o_solution_array = [t, xp[t].solution_value(), xs[t].solution_value(), gd[t].solution_value()]
        other_solution_set = np.vstack((other_solution_set, o_solution_array))
    print(other_solution_set)
    '''
    # Print Merged Solution Set
    merged_solution_set = ['Period', 'SoC', 'Charge', 'Discharge', 'Reserve (for Uncertainty)', 'Purchase', 'Sell',
                           'Diesel', 'Cost']
    for t in range(options_p.time_horizon):
        # Calculate cost per period
        cost_per_period = 0
        # Trade #
        if options_p.option_purchase == 1:
            cost_per_period += (xp[t].solution_value() * real_p.price_purchase[t])
        if options_p.option_sell == 1:
            cost_per_period += (xs[t].solution_value() * -1 * real_p.price_sell[t])
        # Generation #
        # Diesel
        if options_p.option_diesel == 1:
            for j in range(model_p.dg_speed_level):
                cost_per_period += (gdl[j, t].solution_value() * model_p.price_diesel[j])
        # Battery Speed #
        for j in range(model_p.battery_speed_level):
            cost_per_period += (s[j, t].solution_value() * model_p.price_battery[j])
        # Degradation #
        b = model_p.battery_replacement_cost / \
            (model_p.battery_lifetime_throughput * math.sqrt(model_p.round_trip_efficiency))
        if options_p.degradation_unit == 1:
            cost_per_period += (yc[t].solution_value() * b)

        m_solution_array = [t, l[t].solution_value(), yc[t].solution_value(), yd[t].solution_value(),
                            lf[t].solution_value(), xp[t].solution_value(), xs[t].solution_value(),
                            gd[t].solution_value(), cost_per_period]
        merged_solution_set = np.vstack((merged_solution_set, m_solution_array))
    print(merged_solution_set)

    return yc[0].solution_value(), yd[0].solution_value(), merged_solution_set


# Run ScheduleBattery()
# ScheduleBattery(1, 1, 1, 1)


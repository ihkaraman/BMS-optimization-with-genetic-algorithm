# Import Libraries
import pandas as pd
import numpy as np
import math


class ReadParameters(object):
    def __init__(self):

        # Time Horizon (L - array size)
        self.time_horizon = 120
        # Consumption
        # Talep
        self.consumption = pd.read_csv("miris_load_hour.csv")
        self.consumption = list(self.consumption['Conso'])
        # Solar Generation
        self.generation_solar = pd.read_csv("miris_PV_Hour.csv")
        self.generation_solar = list(self.generation_solar['PV'])
        # Initial Load (L)
        self.initial_load = 0

        # Battery Technical Characteristics
        self.battery_capacity_max = 130000
        self.battery_capacity_min = 0
        self.round_trip_efficiency = 0.9
        self.battery_replacement_cost = 100
        self.battery_lifetime_throughput = 647400

        # Unit battery cost
        self.unit_battery_cost = 1000

        # Price
        self.price_purchase = 1000
        self.price_sell = 1000


def CalculateFitness(l_array):

    param = ReadParameters()

    b = np.max(l_array)
    charge = []
    discharge = []
    sell = []
    purchase = []

    for i in range(param.time_horizon):
        # Charge and Discharge
        if i == 0:
            if param.initial_load < l_array[0]:
                charge.append(l_array[0])
                discharge.append(0)
            else:
                charge.append(0)
                discharge.append(0)
        else:
            if l_array[i-1] <= l_array[i]:
                charge.append(l_array[i] - l_array[i-1])
                discharge.append(0)
            else:
                charge.append(0)
                discharge.append(l_array[i-1] - l_array[i])
        # Sell & Purchase
        need = param.consumption[i] + charge[i] - param.generation_solar[i] - discharge[i]
        if need <= 0:
            sell.append(-1 * need)
        else:
            purchase.append(need)

    obj = 0
    # Calculate Objective
    b_cost = param.battery_replacement_cost / (param.battery_lifetime_throughput *
                                                 math.sqrt(param.round_trip_efficiency))
    obj += param.unit_battery_cost * b
    obj += b_cost * np.sum(charge)
    obj += param.price_sell * np.sum(sell)
    obj += param.price_purchase * np.sum(purchase)

    return obj / 1000000


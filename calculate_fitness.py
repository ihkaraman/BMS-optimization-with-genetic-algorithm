# Import Libraries
import pandas as pd
import numpy as np
import math


class CalculateFitness(object):
    
    def __init__(self, data):

        # Consumption
        self.consumption = list(pd.read_csv(data['load'])['Conso'])
        # Solar Generation
        self.generation_solar = list(pd.read_csv(data['pv'])['PV'])
        # Initial Load (L)
        self.initial_load = 0
        
        # Time Horizon (L - array size)
        self.time_horizon = len(self.consumption)

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


    def fitness(self, l_array):
            
        b = np.max(l_array)
        charge = []
        discharge = []
        sell = []
        purchase = []
    
        for i in range(self.time_horizon):
            # Charge and Discharge
            if i == 0:
                if self.initial_load < l_array[0]:
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
            need = self.consumption[i] + charge[i] - self.generation_solar[i] - discharge[i]
            if need <= 0:
                sell.append(-1 * need)
            else:
                purchase.append(need)
    
        obj = 0
        # Calculate Objective
        b_cost = self.battery_replacement_cost / (self.battery_lifetime_throughput *
                                                     math.sqrt(self.round_trip_efficiency))
        obj += self.unit_battery_cost * b
        obj += b_cost * np.sum(charge)
        obj += self.price_sell * np.sum(sell)
        obj += self.price_purchase * np.sum(purchase)
    
        return obj / 1000000


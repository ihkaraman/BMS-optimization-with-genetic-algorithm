import time
import numpy as np

import kmeans
from calculate_fitness import CalculateFitness
import matplotlib.pyplot as plt


class GeneticAlgorithm():
    
    def __init__(self, parameters):
        
          data, pop_size, k, fitness_func,  cros_type, cros_probability, elit_ratio, max_parent_all, \
          mut_prob, mut_rate, max_iter, ind_size, stop_thrs, mut_level = parameters
          
          self.data  = data
          
          self.pop_size = pop_size
          self.k = k
          self.fitness_func = fitness_func
          self.cros_type = cros_type
          self.crossover_prob = cros_probability
          self.elitism_ratio = elit_ratio
          self.max_parent_allow = max_parent_all
          self.mut_prob = mut_prob
          self.mut_rate = mut_rate
          self.max_iter = max_iter
          self.ind_size = ind_size
          self.stop_thrs = stop_thrs
          self.m_level = mut_level
          
          self.fitness_values_dict = {}
          self.solution_individuals = []
          self.solution_costs = []
          self.run_time = None
          self.results = {}
          self.global_min_cost = None 
          self.global_best_solution = None
        
          self.cf = CalculateFitness(self.data)


    def calculate_fitness(self, individual):
           
        def fitness(key):
            
            try:
                self.fitness_values_dict[key] = self.cf.fitness(individual)
            except:
                return 999999.9 # if a solution is not feasible then assign its obj func
            return self.fitness_values_dict[key]
        
        # key = str(list(individual.ravel()))
        key = str(individual)
        
        if key in self.fitness_values_dict:
            return self.fitness_values_dict[key]
        else:
            return fitness(key)


    def generate_random_array(self):
        
        import pandas as pd
          
        data_consumption= pd.read_csv(self.data['load'], usecols=['Rounded'])
        data_generation_solar = pd.read_csv(self.data['pv'], usecols=['Rounded'])
            
        max_consumption = data_consumption["Rounded"].max()
        max_generation_solar = data_generation_solar["Rounded"].max()
        
        upper_limit = max(max_consumption, max_generation_solar)
        lower_limit = 0 
        
        random_array = np.random.randint(lower_limit, upper_limit,  size = (self.pop_size*self.k, self.ind_size))
        
        return random_array


    def initialize_population(self):
    
        points = self.generate_random_array()
        clusters, centroids = kmeans.kmeans(points, self.k)
    
        initial_population = []
    
        for i in range(1,self.k+1):
    
            tmp_points = points[clusters==i]
            costs = [self.cf.fitness(point) for point in tmp_points]
            max_idx = min(len(costs), self.pop_size//self.k)
            best_indexes = sorted(range(len(costs)), key=lambda i: costs[i])[:max_idx]
            [initial_population.append(tmp_points[idx]) for idx in best_indexes]
    
        if len(initial_population) < self.pop_size:
            initial_population = np.append(initial_population, self.generate_random_array(self.pop_size-len(initial_population), self.ind_size), axis=0)
        
        return np.array(initial_population)


    def create_mating_pool(self, population, choice_size=2):
            
        if self.fitness_func == 'distance':
            
            # calculating choice probabilities of a population based on total distance of an ind to others
            def calculate_probabilities(population):
    
                # calculating euclidean distance between two individual
                def calculate_distance(x, y):
                    return np.sqrt(sum((x-y)**2))
    
                # import kmeans
                # fitness_values = [sum([kmeans.calculate_distance(ind1, ind2) for ind2 in population]) for ind1 in population]        
                fitness_values = [sum([calculate_distance(ind1, ind2) for ind2 in population]) for ind1 in population]
                total_fitness = sum(fitness_values)+0.00000001
                choice_probabilities = [fv/total_fitness for fv in fitness_values]   
    
                return choice_probabilities
    
        else:
            
            # calculating choice probabilities of a population
            def calculate_probabilities(population):
    
                fitness_values = [1/self.cf.fitness(individual) for individual in population]
                total_fitness = sum(fitness_values)
                choice_probabilities = [fv/total_fitness for fv in fitness_values]   
                
                return choice_probabilities  
        
        elite_solutions = []
    
        # if elitism raio is greater than 0, add elite solutions to mating pool
        if self.elitism_ratio > 0:
            
            # determining elite solutions
            elite_individual_number = int(self.elitism_ratio*self.pop_size)
            costs = [self.cf.fitness(individual) for individual in population]
            # take the individuals that has least cost values (minimization)
            best_individual_indexes = sorted(range(len(costs)), key=lambda i: costs[i])[:elite_individual_number] 
            elite_solutions = [population[elt] for elt in best_individual_indexes]
        
        mating_pool = [] 
        # each individual can be a parent for limited times
        times_parent = np.zeros(self.pop_size) 
    
        # in each step 'choice_size' of parents will be chosen for number of steps
        while len(mating_pool) < (self.pop_size-len(elite_solutions))//choice_size:
            
            # drawing 'choice_size' random uniform numbers
            draws = np.random.uniform(0, 1, choice_size)
            # index array to control parent eligibility in a population
            idx_array = np.array(range(self.pop_size))
            # an individual can be a parent if it didn't chosen as parent 'max_parent_allow*pop_size' times
            idx_array = idx_array[times_parent<int(self.max_parent_allow*self.pop_size)]

            parent_indexes = []
            # choosing an individual according to the draw probability
            for draw in draws:
               
                accumulated = 0
                
                for idx, probability in zip(idx_array, calculate_probabilities(population[idx_array])):
                    
                    accumulated += probability
                    if draw <= accumulated:
                        parent_indexes.append(idx)
                        idx_array = idx_array[idx_array!=parent_indexes]
                        break
            
            # if the mate is not exist in the mating pool add it
            if parent_indexes not in mating_pool and parent_indexes[::-1] not in mating_pool:           
    
                mating_pool.append(parent_indexes)
                times_parent[parent_indexes] += 1
        
        return np.array(elite_solutions), np.array([population[mate] for mate in mating_pool], dtype=object)
    
    
    def crossover_operator(self, parents):
        
        # make crossover with probability of crossover_probability else return parents
        draw = np.random.uniform(0, 1)
        
        if draw > self.crossover_prob:
            return parents
    
        parent_1, parent_2 = parents
            
        # choosing random two points between 1 and size of individual to determine cutpoints
        random_points = list(np.random.choice(np.arange(1,self.ind_size-1), self.cros_type, replace=False)) 
        
        # creating start and end points of cuts
        cutpoints = sorted(random_points + [0, self.ind_size])
        cut_pieces = {cutpoints[i]:cutpoints[i+1] for i in range(len(cutpoints)-1)}
        
        # cutting parents from the given cut points
        parent_pcs = [(parent_1[start:end], parent_2[start:end]) for start, end in cut_pieces.items()]
        
        # crossover parents from cutpoints, n-point crossover is used (exchanging pieces from different parents)
        mask = 0
        
        offspring_1 = np.array([], dtype=int)
        offspring_2 = np.array([], dtype=int)
        
        for pcs1, pcs2 in parent_pcs:
            
            if mask%2 == 0:
                offspring_1 = np.append(offspring_1, pcs1)
                offspring_2 = np.append(offspring_2, pcs2) 
            else:
                offspring_1 = np.append(offspring_1, pcs2)
                offspring_2 = np.append(offspring_2, pcs1)
            
            mask +=1
            
        return np.array([offspring_1, offspring_2])
    

    def mutation_operator(self, offsprings):
        
        mutated_offsprings = []
        
        # the number of genes will be mutated
        mutation_size = int(self.mut_rate*self.ind_size)
        
        for offspring in offsprings:
            
            # mutate individuals with 'mutation_probability'
            if np.random.uniform(0, 1) < self.mut_prob:
                
                # choose some random number to decide which genes will be mutated
                mutation_mask = np.random.choice(range(self.ind_size), size=mutation_size, replace=False)
                # determine whether or not a gene will increase or decrease by 'mutation_level' 
                mutation_multiplier = self.m_level*np.random.choice([-1,1], size=mutation_size, replace=True) + 1
                # updating genes using'mutation_mask' by 'mutation_multiplier'
                offspring[mutation_mask] = offspring[mutation_mask]*mutation_multiplier
                mutated_offsprings.append(offspring)
            
            else:
                mutated_offsprings.append(offspring)
                
        return np.array(mutated_offsprings)


    def stop_algorithm(self, iter_count):
        
        if iter_count == self.max_iter:
            self.results['stop_cond'] = 'Max iterations'
        else:
            self.results['stop_cond'] = 'Convergence'
        
        self.global_min_cost = min(self.solution_costs)
        min_index = self.solution_costs.index(self.global_min_cost)
        self.global_best_solution = self.solution_individuals[min_index]
        
        self.results['iter num'] = iter_count
        self.results['min_cost'] = self.global_min_cost
        self.results['best_soln'] = self.global_best_solution
        self.results['cost_func'] = self.solution_costs
        self.results['run_time'] = self.run_time


    def print_results(self):
        
        print('---------------------------------------------------')
        print('Algorithm finished...')
        print()
        
        if self.results['stop_cond'] == 'Max iterations':
            print('Stopping condition: Max iterations reached!')
        else:
            print('Stopping condition: Convergence!')
        
        print('----------------')
        print('Best cost function: ', self.results['min_cost'])
        
        print('----------------')
        print('Best individual: ')
        print(self.results['best_soln'])
        print('----------------')  
        print('Algorithm runnig time: ', self.results['run_time'])
        print('----------------')
        
        self.plot_cost()


    def plot_cost(self):
        
        plt.rcParams["figure.figsize"] = (10,6)
        
        x = range(len(self.solution_costs))
        y = self.solution_costs
    
        plt.plot(x, y)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('Cost Function')

    def run(self):
        
        # initialize generation
        start_time = time.time()
        
        generation = self.initialize_population() # array
        iter_count = 0
        
        while iter_count < self.max_iter:
            
            # generating new population with previous population
            prev_generation = generation
            generation_list = []
            # creating mating pool
            elite_solutions, mating_pool = self.create_mating_pool(prev_generation)
            
            # creating new offsprings by crossover and mutation
            for mate in mating_pool: 
    
                offsprings = self.crossover_operator(mate)
                mutated_offsprings = self.mutation_operator(offsprings)
                [generation_list.append(i) for i in mutated_offsprings]
            
            generation = np.append(elite_solutions, np.array(generation_list), axis=0)
            
            # calculating costs of new generation
            cost_values = [self.cf.fitness(individual) for individual in generation]
            # saving min cost and related individual obtained in this iteration
            min_itr_cost = min(cost_values)
            min_itr_index = cost_values.index(min_itr_cost)
            min_itr_individual = generation[min_itr_index]
            
            # if the last n solutions are the same then stop the algorithm
            # if len(solution_costs) > stop_thrs and sum([i==min_itr_cost for i in solution_costs[-stop_thrs:]])>stop_thrs-1:
            #    break  
            self.solution_individuals.append(min_itr_individual)
            self.solution_costs.append(min_itr_cost)
            
            iter_count +=1  
        
        self.run_time = (time.time()-start_time)/60
        self.stop_algorithm(iter_count)
        


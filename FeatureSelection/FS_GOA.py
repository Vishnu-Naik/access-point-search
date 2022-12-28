import numpy as np
from BinaryGOA.GOA import OriginalGOA
# from mealpy.math_based.RUN import OriginalRUN


def fitness_function(solution):
    return np.sum(solution**2)

problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [0],
    "ub": [1.99],
    "minmax": "max",
}

epoch = 1000
pop_size = 50

def call_GOA(problem_dict1, epoch, pop_size):
    c_min = 0.00004
    c_max = 1.0
    model = OriginalGOA(epoch, pop_size, c_min, c_max)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")

call_GOA(problem_dict1, epoch, pop_size)



def call_RUN(problem_dict1, epoch, pop_size):
    model = OriginalRUN(epoch, pop_size)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")

# call_RUN(problem_dict1, epoch, pop_size)
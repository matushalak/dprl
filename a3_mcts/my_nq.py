import numpy as np
from joblib import Parallel, delayed

def initialization_n_queens(population_size, num_of_dims):
    """Generate a population of solutions."""
    population = []
    for _ in range(population_size):
        individual = np.arange(0, num_of_dims)
        np.random.shuffle(individual)
        population.append(individual)
    return population

def evaluate_solution_n_queens(solution):
    """Calculate the fitness of an solution."""
    ################################################################
    # fitness = number of checking queens, only need to check for diagonal
    # represent as matrix, ones are queens
    matrix = np.zeros((len(solution), len(solution)))
    # need separate arrays to directly index np array with row-column pairs
    cols, rows = zip(*enumerate(solution))
    matrix[rows, cols] = 1
    #print(matrix)

    a = matrix.shape[0]
    diagonals = [np.diag(matrix, k=i).sum() for i in range(-a+1,a)]
    antidiagonals = [np.diag(np.fliplr(matrix), k=i).sum() for i in range(-a+1,a)]
    all_diag = diagonals + antidiagonals

    fitness = 0
    for dg in all_diag:
        fitness += dg if dg > 1 else 0
    ################################################################
    
    return fitness

def visualize_solution(solution):
    """Visualize the placement of queens on the chessboard."""

    ################################################################
    size = len(solution)
    printout = ''
    for row in range(size):
        for col in range(size):
            if solution[col] == row:
                printout += ' Q '
            else:
                printout += ' . '
        printout += '\n'

    print(printout)

def evaluation_n_queens(population: list[list]) -> list[int]:
    """Evaluate the whole population and return the fitness of each using joblib.
    !!!! PARALLELIZED !!!! makes everything MUCH faster!!!
    """
    fitnesses = Parallel(n_jobs=-1)(delayed(evaluate_solution_n_queens)(ind) for ind in population)
    return fitnesses

def fitness_sharing(fitnesses: list[int], population: list[list], sigma_share: float) -> list[float]:
    """Apply fitness sharing to promote diversity and niche creation."""
    shared_fitnesses = []
    for i, fitness_i in enumerate(fitnesses):
        niche_count = 0
        for j, individual_j in enumerate(population):
            if i != j:
                distance = np.sum(population[i] != population[j])  # Hamming distance
                if distance < sigma_share:
                    niche_count += (1 - (distance / sigma_share)) ** 2
        shared_fitness = fitness_i / (1 + niche_count)
        shared_fitnesses.append(shared_fitness)
    return shared_fitnesses

# Partially Mapped Crossover (PMX)
def recombine(p1: list, p2: list) -> list:
    size = len(p1)
    cx_point1, cx_point2 = sorted(np.random.choice(np.arange(0, size), 2, replace=False))

    offspring = [-1] * size
    # Copy the crossover segment from p1
    offspring[cx_point1:cx_point2] = p1[cx_point1:cx_point2]

    # Fill the remaining positions using p2
    for i in range(cx_point1, cx_point2):
        if p2[i] not in offspring:
            j = i
            while cx_point1 <= j < cx_point2:
                j_positions = np.where(p1 == p2[j])[0]
                if len(j_positions) > 0:
                    j = j_positions[0]
                else:
                    break  # Break out if no valid index is found
            if j < size and offspring[j] == -1:
                offspring[j] = p2[i]

    # Fill any remaining -1 values with the rest of p2, ensuring no duplicates
    used_values = set(offspring)  # Track used values
    for i in range(size):
        if offspring[i] == -1:
            for val in p2:
                if val not in used_values:
                    offspring[i] = val
                    used_values.add(val)
                    break

    return offspring

# simple cut and crossfill crossover
# def recombine(p1:list, p2:list)->list:
#         # pivot point
#         pivot = np.random.randint(len(p1))
#         segment1 = p1[:pivot]
#         segment2 = []
#         segment2_temp = np.concatenate((p2[pivot:],p2[:pivot])) 
#         for pos in (segment2_temp):
#             if pos not in segment1:
#                 segment2.append(pos)
#         return np.concatenate((segment1, segment2))

def crossover_n_queens(x_parents, p_crossover):
    """Perform PMX crossover to create offsprings."""
    offspring = []
    for parent1, parent2 in zip(x_parents[::2], x_parents[1::2]):
        if np.random.uniform() > p_crossover:
            offspring.append(parent1)
            offspring.append(parent2)
        else:
            child1 = recombine(parent1, parent2)
            child2 = recombine(parent2, parent1)
            offspring.append(child1)
            offspring.append(child2)
    return offspring

# Mutation with inversion and scramble
def mutation_n_queens(individual, mutation_rate):
    """Apply mutation (swap, inversion, or scramble) to an individual."""
    if np.random.uniform() > mutation_rate:
        return individual
    else:
        individual_mutated = individual.copy()
        if np.random.uniform() < 0.5:
            # Swap mutation
            i, j = np.random.choice(np.arange(0, len(individual)), 2, replace=False)
            individual_mutated[i], individual_mutated[j] = individual_mutated[j], individual_mutated[i]
        else:
            # Inversion mutation
            i, j = sorted(np.random.choice(np.arange(0, len(individual)), 2, replace=False))
            individual_mutated[i:j] = np.flip(individual[i:j])
        return individual_mutated

# Tournament selection
def tournament_selection(population, shared_fitnesses, k: int = 8) -> list:
    """Tournament selection using shared fitness."""
    players = np.random.choice(np.arange(len(population)), size=k)
    best_individual_idx = min(players, key=lambda i: shared_fitnesses[i])
    return population[best_individual_idx]

def parent_selection_n_queens(population, fitnesses, selection_function, fitness_function, n_children=2):
    """Select parents for the next generation."""
    n_parents = int(len(population) / n_children)
    x_parents, f_parents = [], []
    for _ in range(n_parents):
        for _ in range(n_children):
            parent = selection_function(population, fitnesses)
            x_parents.append(parent)
            f_parents.append(fitness_function(parent))
    return x_parents, f_parents

# Survivor selection with elitism
def survivor_selection_n_queens(population, fitnesses, offspring, offspring_fitnesses):
    """Select survivors using elitism."""
    total_population = population + offspring
    total_fitnesses = fitnesses + offspring_fitnesses
    elite_count = len(population) // 10
    best_individuals = sorted(zip(total_fitnesses, total_population), key=lambda x: x[0])[:len(population)]
    return [ind for _, ind in best_individuals], [fit for fit, _ in best_individuals]

#%%
def ea_n_queens(population_size, max_generations, p_crossover, m_rate, num_of_dims):
    """Evolutionary algorithm for N-Queens problem."""
    sigma_share = int(0.2 * num_of_dims)    
    # max_generations = int(max_fit_evals / population_size)

    # Initialize population and calculate fitness
    population = initialization_n_queens(population_size, num_of_dims)
    fitnesses = evaluation_n_queens(population)

    # Best individual tracking
    idx = np.argmin(fitnesses)
    best_individual = population[idx]
    best_fitness = fitnesses[idx]
    
    # stagnation workaround
    stagnation = 0
    starting_mutation_rate, starting_crossover_rate = m_rate, p_crossover
    for i in range(max_generations):
        # Apply fitness sharing for niching
        shared_fitnesses = fitness_sharing(fitnesses, population, sigma_share)

        # Parent selection
        parents, fitnesses = parent_selection_n_queens(population, shared_fitnesses, tournament_selection, evaluate_solution_n_queens)

        # Crossover and mutation
        offspring = crossover_n_queens(parents, p_crossover)
        offspring = [mutation_n_queens(child, m_rate) for child in offspring]
        offspring_fitnesses = evaluation_n_queens(offspring)

        # Survivor selection with elitism
        population, fitnesses = survivor_selection_n_queens(population, fitnesses, offspring, offspring_fitnesses)

        # Check for best solution
        idx = np.argmin(fitnesses)
        if fitnesses[idx] < best_fitness:
            best_fitness = fitnesses[idx]
            best_individual = population[idx]
            stagnation = 0 # reset stagnation
            m_rate, p_crossover = starting_mutation_rate, starting_crossover_rate

        else:
            stagnation += 1
            if stagnation < 10:
                m_rate += .02
                p_crossover += 0.03
            elif stagnation >= 10 and stagnation < 20:
                m_rate += .03
                p_crossover += 0.05
            else:
                # too long stagnation, need new blood
                new_blood = initialization_n_queens(population_size//3, num_of_dims)
                new_fitnesses = evaluation_n_queens(new_blood)

                # replace a third of population with new blood
                population[-(population_size // 3):] = new_blood
                fitnesses[-(population_size // 3):] = new_fitnesses
    
                stagnation = 0 # reset stagnation
                m_rate, p_crossover = starting_mutation_rate, starting_crossover_rate
                print('-----New Blood!-----')

        print(f'Generation: {i}, best fit: {best_fitness}')
        
        # Early stopping condition
        if best_fitness == 0:
            print(f"Stopping early as solution with fitness 0 found in generation {i}.")
            break

    # End of generations loop without finding a solution with fitness 0
    if best_fitness != 0:
        print(f"Reached max generations. Best solution has fitness {best_fitness}.")

    return best_individual, best_fitness

#%%
if __name__ == '__main__':
    import argparse
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Solve N-Queens problem using evolutionary algorithm")

    # Define arguments
    parser.add_argument('-p', '--popsize', type=int, required=False, default = 100, help="Population size")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 500, help="Max generations")
    parser.add_argument('-cr', '--crossover_rate', type=float, required=False, default = 0.5, help="Crossover rate (e.g., 0.8)")
    parser.add_argument('-mr', '--mutation_rate', type=float, required=False, default = 0.1, help="Mutation rate (e.g., 0.05)")
    parser.add_argument('-nq', '--nqueens', type=int, required=False, help="Number of Queens (e.g., N=8)")

    # Parse the arguments
    args = parser.parse_args()

    # Now use the parsed arguments in your program logic
    if isinstance(args.nqueens, int):
        nq = args.nqueens
    else:
        nq = int(input("Enter Number of Queens:")) #say N = 8

    popsize = args.popsize
    mg = args.maxgen
    cr = args.crossover_rate
    mr = args.mutation_rate

    print(f"Case when N = {nq}:")
    x_best, f_best = ea_n_queens(popsize, mg, cr, mr, nq)

    print(f"For N = {nq}:")
    print("Best fitness:", f_best)
    print("Best solution found:")
    print(x_best) # so that its possible to check
    visualize_solution(x_best)

"""
Generation: 344, best fit: 0
Stopping early as solution with fitness 0 found in generation 344.
For N = 100:
Best fitness: 0
Best solution found:
[np.int64(61), np.int64(38), np.int64(52), np.int64(31), np.int64(83), np.int64(27), np.int64(29), np.int64(56), np.int64(14), np.int64(51), 
np.int64(67), np.int64(30), np.int64(33), np.int64(53), np.int64(17), np.int64(10), np.int64(62), np.int64(3), np.int64(19), np.int64(46), 
np.int64(80), np.int64(88), np.int64(22), np.int64(32), np.int64(79), np.int64(4), np.int64(73), np.int64(57), np.int64(60), np.int64(68), 
np.int64(96), np.int64(9), np.int64(95), np.int64(89), np.int64(36), np.int64(39), np.int64(20), np.int64(25), np.int64(34), np.int64(50), 
np.int64(71), np.int64(18), np.int64(86), np.int64(94), np.int64(24), np.int64(1), np.int64(5), np.int64(45), np.int64(72), np.int64(59), 
np.int64(75), np.int64(16), np.int64(43), np.int64(82), np.int64(87), np.int64(90), np.int64(74), np.int64(0), np.int64(21), np.int64(93), 
np.int64(26), np.int64(77), np.int64(98), np.int64(8), np.int64(76), np.int64(54), np.int64(58), np.int64(49), np.int64(23), np.int64(63), 
np.int64(37), np.int64(78), np.int64(85), np.int64(12), np.int64(44), np.int64(35), np.int64(28), np.int64(6), np.int64(2), np.int64(15), 
np.int64(13), np.int64(42), np.int64(99), np.int64(97), np.int64(81), np.int64(11), np.int64(48), np.int64(92), np.int64(41), np.int64(70), 
np.int64(66), np.int64(64), np.int64(91), np.int64(40), np.int64(84), np.int64(69), np.int64(65), np.int64(47), np.int64(55), np.int64(7)]
"""
from SoftRobot import *

import argparse
import random
import copy
import json
import itertools
import numpy as np


# Constants defining tissue types
MUSCLE_A = -1
MUSCLE_B = 1
BONE = 0
LIGAMENT = 2
NONE = 3


def random_genotype():
    """
    Creates and returns a random genotype (your choice of randomness).
    The genotype *must* be a 64-element list of BONE, LIGAMENT, MUSCLE_A, MUSCLE_B, and NONE
    """
    tissues = [BONE, LIGAMENT, MUSCLE_A, MUSCLE_B, NONE]
    randomlist = random.choices(tissues, k=64)
    #print(randomlist)
    return randomlist


def mutation(genotype):
    """
    Creates and returns a *new* genotype by mutating the argument genotype.
    The argument genotype must remain unchanged.
    """
    mutation = copy.deepcopy(genotype)
    tissues = [BONE, LIGAMENT, MUSCLE_A, MUSCLE_B, NONE]
    for i in range(len(mutation)):
        if random.randint(1,64) %(i+1) == 0:
            typeT = mutation[i]
            newtype = random.choices(tissues, k=1)[0]
            while typeT == newtype:
                newtype = random.choices(tissues, k=1)[0]
            mutation[i] = newtype
            
    return mutation
    


def crossover(genotype1, genotype2):
    """
    Creates and returns TWO *new* genotypes by applying crossover to the argument genotypes.
    The argument genotypes must remain unchanged.
    """
    
    genotype3 = genotype1.copy()
    temp = genotype1.copy()
    genotype4 = genotype2.copy()
    
    genotype3[0:15] = genotype4[0:15]
    genotype3[48:63] = genotype4[48:63]
    genotype4[0:15] = temp[0:15]
    genotype4[48:63] = temp[48:63]
    
    return genotype3, genotype4


def selection_and_offspring(population, fitness):
    """
    Arguments
        population: list of genotypes
        fitness: list of fitness values (higher is better) corresponding to each
                 of the genotypes in population

    Returns
       A new population (list of genotypes) that is the SAME LENGTH as the 
       argument population, created by selecting genotypes from the argument
       population by their fitness and applying genetic operators (mutation,
       crossover, elitism) 
    """ 
    population2 = population.copy()
    meanfit = np.mean(fitness) 
    for i in range(len(population)):
        if fitness[i] >= meanfit:
            population2.append(population[i])
    poplen = len(population2)
    temp2 = population2.copy()
    while len(population2) < len(population):
        op = random.randint(0,1)
        if (op == 0):
            geno = random.choices(population, k=1)
            population2.append(mutation(geno))
        else:
            genos1 = random.choices(temp2, k =1)
            genos2 = random.choices(population, k =1)
            g1, g2 = crossover(genos1, genos2)
            population2.append(g1)
            if len(population2) < len(population):
                break
            population2.append(g2)
    #print(population2)
    tmp = population2.copy()
    for i in range(len(tmp)):
        population2[i] = mutation(tmp[i])
    return population2


def evolve(num_generations, pop_size, fps, timesteps):
    """
    Runs the evolutionary algorithm. You do NOT need to modify this function, 
    but you should understand what it does.
    """
    for g in range(num_generations):

        if g == 0:
            population = [random_genotype() for p in range(pop_size)]
        else:
            population = selection_and_offspring(population, fitness)

        fitness = []
        for i,p in enumerate(population):
            simulation = SoftRobot(p, f=fps)
            render = True if g%10 == 0 else False
            fit = simulation.Run(timesteps, render)
            fitness.append(fit)
            print(f"Individual {i} has fitness {fit}")

        print(f"\tGeneration {g+1}: Highest Fitness {max(fitness)}")

        
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run soft robot evolution")
    parser.add_argument("num_generations", type=int, help="Number of generations (int)")
    parser.add_argument("-p", "--pop_size", type=int, default=10, help="Population size (int)")
    parser.add_argument("-t", "--timesteps", type=int, default=300, help="Timesteps per simulation (int)")
    parser.add_argument("-f", "--fps", type=int, default=60, help="Frames per second for simulation (int)")
    args = parser.parse_args()

    # Run evolution
    evolve(args.num_generations, args.pop_size, args.fps, args.timesteps)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
SARS-CoV-2 probabilistic simulator
https://github.com/covid19-modeling

This simulator was used to train the predictions available at:
https://covid19-modeling.github.io/

The method is described in detail in the preprint:

    Matabuena, M., Meijide-García, C., Rodríguez-Mier, P., & Leborán, V. (2020).
    COVID-19: Estimating spread in Spain solving an inverse problem with a probabilistic model.
    arXiv preprint arXiv:2004.13695. https://arxiv.org/abs/2004.13695

@author: Pablo Rodríguez-Mier (@pablormier)
"""

import numpy as np
import multiprocessing as mp
import warnings
from collections import namedtuple
from enum import IntEnum, unique


RNG = np.random.Generator(np.random.PCG64(np.random.SeedSequence()))
Model = namedtuple('Model', ['transitionMatrix', 'timeSimulator', 'parameters', 'alphas', 'betas', 'means'])
Schedule = namedtuple('Schedule', ['incoming', 'outgoing'])

# TODO: Future versions will incorporate a way to define custom Markov Chains
# different for the one we used to model SARS-CoV-2

STATE_NAMES = [
    'I1: Infected (incubating)',
    'I2: Infected (asymptomatic)',
    'I3: Infected (symptomatic)',
    'R1: Recovered (infectious)',
    'R2: Removed',
    'M0: Dead']


# States of the Markov Model
@unique
class States(IntEnum):
    I1 = 0
    I2 = 1
    I3 = 2
    R1 = 3
    R2 = 4
    M0 = 5


S = States

DYN_RI_DEFAULT_PARAMS = [
    1.500,
    0.500,
    9.000,
    0.003,
]

MARKOV_DEFAULT_PARAMS = [
    0.8,    # alpha: probability of going from I1 (infected) to I3 (symptomatic)
    0.06,   # beta: probability of going from I3 (symptomatic) to dead
    5.807,  # Gamma1 shape (time from I1 to I2)
    0.948,  # Gamma1 scale
    5.807,  # Gamma2 shape (time from I1 to I3)
    0.948,  # Gamma2 scale
    6.670,  # Gamma3 shape (time from I3 to M)
    2.550,  # Gamma3 scale
    9.,     # Uniform1 lower bound to model transition times from I3 to R1
    14.,    # Uniform1 upper bound
    5.,     # Uniform2 lower bound to model transition times from I2 to R1
    10.,    # Uniform2 upper bound
    7.,     # Uniform3 lower bound to model transition times from R1 to R2
    14.     # Uniform4 upper bound
]

DEFAULT_PARAMS = DYN_RI_DEFAULT_PARAMS + MARKOV_DEFAULT_PARAMS


def infectious(state):
    return int(state[S.I1] + state[S.I2] + state[S.I3] + state[S.R1])


def default_rit_function(day, params):
    if len(params) != 4:
        raise ValueError("Unexpected number of parameters for R0(t) function")
    a, b, c, d = params
    return d + a / (1 + b ** (-(day - c)))


def infection_rates(params, num_days=60):
    return [default_rit_function(i, params) for i in range(num_days)]


def create_transition_matrix(alpha, beta):
    num_states = len(STATE_NAMES)
    transitionMatrix = np.zeros(shape=(num_states, num_states))
    transitionMatrix[S.I1, S.I3] = 1 - alpha
    transitionMatrix[S.I1, S.I2] = alpha
    transitionMatrix[S.I2, S.R1] = 1
    transitionMatrix[S.I3, S.M0] = beta
    transitionMatrix[S.I3, S.R1] = 1 - beta
    transitionMatrix[S.R1, S.R2] = 1
    return transitionMatrix


def build_markovchain(params, alphas=None, betas=None):
    num_states = len(STATE_NAMES)
    # TODO: Refactor to use a MarkovChain class instead. 
    # From v0.1.6, alpha/beta can be a daily list of values, that
    # can be pased to the namedtuple. In that case, alpha and beta
    # from params are simply ignored. This is a temporary solution. 
    alpha = params[0]
    beta = params[1]
    pGamma_I1_I2 = params[2:4]
    pGamma_I1_I3 = params[4:6]
    pGamma_I3_M = params[6:8]
    pUniform_I3_R1 = params[8:10]
    pUniform_I2_R1 = params[10:12]
    pUniform_R1_R2 = params[12:14]

    # Each transition has a distribution function to generate the times when
    # the transitions are going to be made.
    # TODO: Replace by a dict
    distributions = np.full(shape=(num_states, num_states), fill_value=None)
    distributions[S.I1, S.I2] = lambda amount, rng=RNG: np.ceil(
        rng.gamma(pGamma_I1_I2[0], scale=pGamma_I1_I2[1], size=amount))
    distributions[S.I1, S.I3] = lambda amount, rng=RNG: np.ceil(
        rng.gamma(pGamma_I1_I3[0], scale=pGamma_I1_I3[1], size=amount))
    distributions[S.I3, S.M0] = lambda amount, rng=RNG: np.ceil(
        rng.gamma(pGamma_I3_M[0], scale=pGamma_I3_M[1], size=amount))
    distributions[S.I3, S.R1] = lambda amount, rng=RNG: rng.integers(
        low=pUniform_I3_R1[0], high=pUniform_I3_R1[1], size=amount, endpoint=True)
    distributions[S.I2, S.R1] = lambda amount, rng=RNG: rng.integers(
        low=pUniform_I2_R1[0], high=pUniform_I2_R1[1], size=amount, endpoint=True)
    distributions[S.R1, S.R2] = lambda amount, rng=RNG: rng.integers(
        low=pUniform_R1_R2[0], high=pUniform_R1_R2[1], size=amount, endpoint=True)

    # Quick patch: compute the mean of the distribution by sampling.
    # Assume that the user can provide any type of function in the future.
    samples = 10000
    means = np.zeros(shape=(num_states, num_states), dtype=int)
    means[S.I1, S.I2] = np.mean(distributions[S.I1, S.I2](samples)).astype(int)
    means[S.I1, S.I3] = np.mean(distributions[S.I1, S.I3](samples)).astype(int)
    means[S.I3, S.M0] = np.mean(distributions[S.I3, S.M0](samples)).astype(int)
    means[S.I3, S.R1] = np.mean(distributions[S.I3, S.R1](samples)).astype(int)
    means[S.I2, S.R1] = np.mean(distributions[S.I2, S.R1](samples)).astype(int)
    means[S.R1, S.R2] = np.mean(distributions[S.R1, S.R2](samples)).astype(int)

    return Model(create_transition_matrix(alpha, beta), distributions, params, alphas, betas, means)


def simulate_day(day, schedule, model, rng, expectation_mode=False):
    max_days = np.size(schedule.incoming, 0)
    remaining_days = max_days - day
    # If a list of alphas/betas is provided, reconstruct the transition matrix
    if model.alphas is not None and model.betas is not None:
        if len(model.alphas) != len(model.betas):
            raise ValueError("Number of alphas and betas must be equal")
        if len(model.alphas) != max_days or len(model.betas) != max_days:
            raise ValueError("Number of alphas and betas must be equal to the number of days")
        transitionMatrix = create_transition_matrix(model.alphas[day], model.betas[day])
    else:
        transitionMatrix = model.transitionMatrix

    for i in range(np.size(transitionMatrix, 1)):
        destinations = np.where(transitionMatrix[i, :] != 0)[0]
        incoming = int(schedule.incoming[day, i])
        # No destination from here
        if np.size(destinations) == 0:
            continue
        # Nothing to schedule out today
        if incoming == 0:
            continue
        # Get the transition probability for each state
        probs = transitionMatrix[i, destinations].tolist()
        # Optimize this step by multiplying by the probabilities. Add an option to switch to the expectation mode
        average_assignation, assignations = None, None
        if expectation_mode:
            average_assignation = [np.round(probs[i] * incoming).astype(int) for i in range(len(destinations))]
        else:
            assignations = rng.choice(destinations, size=incoming, p=probs, replace=True)
        for j in range(len(destinations)):
            dest = destinations[j]
            # Amount of people designated to state j
            if expectation_mode:
                amount = average_assignation[j]
            else:
                amount = np.sum(assignations == dest)
            if amount == 0:
                continue
            # Get the time simulator from current to destination
            time_dist = model.timeSimulator[i, dest]
            # TODO: If expectation_mode = True, use the mean of the distribution               
            sampled_times = time_dist(amount, rng).astype(int)
            # Update the schedule table to indicate which days they are entering
            counts = np.zeros(remaining_days)
            # If there is only one person, no need to count. We just fill the corresponding day
            if expectation_mode:
                entering_day = model.means[i, dest]
                if entering_day < remaining_days:
                    counts[entering_day] = amount
            elif np.size(sampled_times) <= 1:
                entering_day = sampled_times[0]
                if entering_day < remaining_days:
                    counts[entering_day] = 1
            else:
                c = np.bincount(sampled_times.astype(int))
                # Fill the corresponding days, discarding days ahead
                # the remaining time in the simulation
                s = min(remaining_days, np.size(c))
                counts[:s] = c[:s]
            schedule.outgoing[day:, i] = schedule.outgoing[day:, i] - counts
            schedule.incoming[day:, dest] = schedule.incoming[day:, dest] + counts
    return schedule


def simulation(susceptible_population, initial_infections, model, daily_ri_values, expectation_mode=False, rng=RNG):
    num_states = len(STATE_NAMES)
    num_days = len(daily_ri_values)
    total_population = susceptible_population + initial_infections
    matrix_size = (num_days, num_states)
    state_counts = np.zeros(shape=matrix_size)
    schedule = Schedule(np.zeros(shape=matrix_size), np.zeros(shape=matrix_size))
    for i in range(num_days - 1):
        # Inject new infections into I1 each day in the main loop
        if i == 0:
            new_infections = initial_infections
        else:
            infectious_amount = infectious(state_counts[i, :])
            lambda_t = daily_ri_values[i] * infectious_amount
            if expectation_mode:
                total = np.round(lambda_t).astype(int)
            else:
                total = rng.poisson(lam=lambda_t, size=1)
            new_infections = max(0, min(susceptible_population, total))
            susceptible_population = susceptible_population - new_infections

        schedule.incoming[i, S.I1] = new_infections
        schedule = simulate_day(i, schedule, model, expectation_mode=expectation_mode, rng=rng)
        state_counts[i + 1, :] = state_counts[i, :] + schedule.incoming[i, :] + schedule.outgoing[i, :]
        total = susceptible_population + np.sum(state_counts[i + 1, :])
        if total != total_population:
            raise ValueError("Unexpected invalid state at day {}".format(i))
    return state_counts, schedule


def sample_chains(susceptible, initial_infected, model, daily_ri_values, num_chains=1000,
                  n_workers=None, pool=None, expectation_mode=False, show_progress=False):

    if n_workers is not None and n_workers > 1 and pool is None:
        pool = initialize_pool(n_workers, np.random.SeedSequence())

    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm as tq
            pbar = tq(total=num_chains)
        except:
            warnings.warn("Could not load tqdm to show progress, please install it to use the option show_progress")

    simulations = np.zeros(shape=(num_chains, len(daily_ri_values), len(STATE_NAMES)))
    if pool is None:
        it = (simulation(susceptible, initial_infected, model, daily_ri_values, expectation_mode=expectation_mode) for _ in range(num_chains))
    else:
        it = pool.imap_unordered(_fn_simulation, [(susceptible, initial_infected, model.parameters, daily_ri_values, model.alphas, model.betas, expectation_mode)
                                                  for _ in range(num_chains)])

    for i, (st, _) in enumerate(it):
        simulations[i, :, :] = st
        if pbar is not None:
            pbar.update()

    if pbar is not None:
        pbar.close()

    if pool is not None:
        pool.terminate()
    return simulations


def _fn_simulation(args):
    # Inject the random generator and build the model to allow pickle problems
    # with the lambdas
    s, i, p, v, a, b, e = args
    m = build_markovchain(p, alphas=a, betas=b)
    states, schedule = simulation(s, i, m, v, expectation_mode=e, rng=_fn_simulation.random_generator)
    return states, schedule


def _process_initializer(fn, q):
    fn.random_generator = q.get()


def initialize_pool(n_workers, seed_seq):
    seeds = seed_seq.spawn(n_workers)
    q = mp.SimpleQueue()
    for s in seeds:
        q.put(np.random.Generator(np.random.PCG64(s)))
    # Pass the generators
    return mp.Pool(processes=n_workers, initializer=_process_initializer, initargs=(_fn_simulation, q))

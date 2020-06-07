# Pyncov-19: Learn and predict the spread of COVID-19
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LzTsXcisv_v_w4q6o9GxvuAzToExFGaG?usp=sharing)

Pyncov-19 is a tiny probabilistic simulator for SARS-CoV-2 implemented in Python 3, whose only dependency is Numpy 1.18.
This simulator is used to learn and predict the temporal dynamics of COVID-19 that are shown in https://covid19-modeling.github.io. It implements a probabilistic compartmental model at the individual level using a Markov Chain model with temporal transitions that were adjusted using the most recent scientific evidence.

## Quick Start

Installation using pip:

```bash
pip install pyncov
```

Sampling 1000 simulated trajectories of the SARS-CoV-2 spread in Madrid:

```python
import pyncov as nc

susceptible = 6680000
infected = 1
num_days = 100

# Those parameters were fitted using Pyncov-19 with CMAES
rit_params = [1.76206245, 0.73465654, 11.46818215, 0.01691976]
# Use the default Ri(t) function with the provided parameters to calculate the daily individual dynamic reproduction rate
daily_ri_values = [nc.default_rit_function(i, rit_params) for i in range(num_days)]

# Instantiate the model with the default parameters and sample 1,000 chains
# NOTE: show_progress requires the TQDM library not installed by default.
m = nc.build_markovchain(nc.MARKOV_DEFAULT_PARAMS)
simulations = nc.sample_chains(susceptible, infected, m, daily_ri_values, 
                               num_chains=1000, n_workers=4, show_progress=True)

```

A more detailed explanation can be found in the notebook included in the repository https://github.com/covid19-modeling/pyncov-19/blob/master/notebooks/basics.ipynb



## About

This library is still a proof-of-concept and it's inteded only to be used for research and experimentation. For more information please read our [preprint](https://arxiv.org/abs/2004.13695):

> Matabuena, M., Meijide-García, C., Rodríguez-Mier, P., & Leborán, V. (2020). 
**COVID-19: Estimating spread in Spain solving an inverse problem with a probabilistic model.**
arXiv preprint arXiv:2004.13695.


This model's main goal is to estimate the levels of infections (or the seroprevalence) of the population, using only data from the registered deaths caused by COVID-19. Although the model can be used to make future predictions (evolution of infections, fatalities, etc.), that's not the primary purpose of the model. Given the uncertainty about essential events that alter the course and dynamics of the spread (for example, the use of masks, lockdowns, social distance, etc.), it is tough to make accurate predictions, so we limit ourselves to use the model to reveal more information about what happened before (backcasting).

# Pyncov-19: Learn and predict the spread of COVID-19
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LzTsXcisv_v_w4q6o9GxvuAzToExFGaG?usp=sharing)

Pyncov-19 is a tiny probabilistic simulator for SARS-CoV-2 implemented in Python 3, whose only dependency is Numpy 1.18.
This simulator is used to learn and predict the temporal dynamics of COVID-19 that are shown in https://covid19-modeling.github.io. It implements a probabilistic compartmental model at the individual level using a Markov Chain model with temporal transitions that were adjusted using the most recent scientific evidence.

## Quick Start

Basic installation using pip with minimal dependencies:

```bash
pip install pyncov
```

In order to install pyncov with the additional features (progress bar, plotting) use:

```bash
pip install pyncov[all]
```

Sampling 100 simulated trajectories of the SARS-CoV-2 spread in Madrid:

```python
import pyncov as nc
import pyncov.io
# Requires pandas
import pyncov.plot

susceptible = 6680000
infected = 1
num_days = 80

parameters = nc.io.get_trained_params('ESP-MD')
# Use the default Ri(t) function with the provided parameters to calculate the daily individual dynamic reproduction rate
values = [nc.default_rit_function(i, parameters) for i in range(num_days)]

# Instantiate the model with the default parameters and sample 100 chains
# NOTE: show_progress requires the TQDM library not installed by default.
m = nc.build_markovchain(nc.MARKOV_DEFAULT_PARAMS)
simulations = nc.sample_chains(susceptible, infected, m, values, 
                               num_chains=100, show_progress=True)

```

You can visualizate the trajectories and the average trajectory matching the observed values in Madrid using `pyncov.plot`:

```python
import matplotlib.pyplot as plt

# Load the dataset with pandas
df = pd.read_csv(...)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
nc.plot.plot_state(simulations, nc.S.I1, ax=ax[0], index=df.index, title="New infections over time")
nc.plot.plot_state(simulations, nc.S.M0, diff=True, ax=ax[1], index=df.index, title="Daily fatalities")
nc.plot.plot_state(simulations, nc.S.M0, ax=ax[2], index=df.index, title="Total fatalities")
df.diff().plot(ax=ax[1]);
df.plot(ax=ax[2]);
```
![](noteboks/assets/madrid_example.png)

A more detailed explanation can be found in the notebook included in the repository https://github.com/covid19-modeling/pyncov-19/blob/master/notebooks/basics.ipynb



## About

This library is still a proof-of-concept and it's inteded only to be used for research and experimentation. For more information please read our [preprint](https://arxiv.org/abs/2004.13695):

> Matabuena, M., Meijide-García, C., Rodríguez-Mier, P., & Leborán, V. (2020). 
**COVID-19: Estimating spread in Spain solving an inverse problem with a probabilistic model.**
arXiv preprint arXiv:2004.13695.


This model's main goal is to estimate the levels of infections (or the seroprevalence) of the population, using only data from the registered deaths caused by COVID-19. Although the model can be used to make future predictions (evolution of infections, fatalities, etc.), that's not the primary purpose of the model. Given the uncertainty about essential events that alter the course and dynamics of the spread (for example, the use of masks, lockdowns, social distance, etc.), it is tough to make accurate predictions, so we limit ourselves to use the model to reveal more information about what happened before (backcasting).

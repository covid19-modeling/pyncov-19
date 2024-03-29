<p align="center">
 <img alt="logo" src="https://avatars.githubusercontent.com/u/64030820?s=200&v=4" height="200">
<h1 align="center" margin=0px>
Pyncov-19: Learn and predict the spread of COVID-19
</h1>
</p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LzTsXcisv_v_w4q6o9GxvuAzToExFGaG?usp=sharing)

Pyncov-19 is a tiny probabilistic simulator for SARS-CoV-2 implemented in Python 3, whose only dependency is Numpy 1.18.
This simulator is used to learn and predict the temporal dynamics of COVID-19 that are shown in https://covid19-modeling.github.io. It implements a probabilistic compartmental model at the individual level using a Markov Chain model with temporal transitions that were adjusted using the most recent scientific evidence.

![](https://github.com/covid19-modeling/pyncov-19/raw/master/notebooks/assets/madrid_example.png)

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

infection_rates = nc.infection_rates(nc.io.get_trained_params('ESP-MD'), num_days=60)
# Instantiate the model with the default parameters and sample 100 chains
# NOTE: show_progress requires the TQDM library not installed by default.
m = nc.build_markovchain(nc.MARKOV_DEFAULT_PARAMS)
simulations = nc.sample_chains(susceptible, infected, m, infection_rates, 
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
![](https://github.com/covid19-modeling/pyncov-19/raw/master/notebooks/assets/madrid_example.png)

A more detailed explanation can be found in the notebook included in the repository https://github.com/covid19-modeling/pyncov-19/blob/master/notebooks/basics.ipynb



## About
The primary objective of this model is to estimate the infection levels (or seroprevalence) of the population based solely on the data from registered deaths attributed to COVID-19. While the model has the capability to project future outcomes (such as infection trends and fatality rates), that isn't its central focus. Given the unpredictability surrounding pivotal events that can shift the trajectory and dynamics of the virus spread—like mask mandates, lockdowns, and social distancing—it's challenging to make precise predictions. Consequently, we primarily use the model for backcasting to shed light on past events.

For more information please read our [paper](https://doi.org/10.1016/j.cmpb.2021.106399):

> Matabuena, M., Rodriguez-Mier, P., Garcia-Meixide, C., & Leboran, V. (2021). COVID-19: Estimation of the transmission dynamics in Spain using a stochastic simulator and black-box optimization techniques. *Computer Methods and Programs in Biomedicine*, 211, 106399.


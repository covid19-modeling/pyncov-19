# Pyncov-19: Simulating the spread of SARS-CoV-2

Pyncov-19 is a tiny probabilistic simulator for SARS-CoV-2 implemented in Python 3, whose only dependency is Numpy 1.18.
This simulator is used to learn and predict the temporal dynamics of COVID-19 that are shown in https://covid19-modeling.github.io. It implements a probabilistic compartmental model at the individual level using a Markov Chain model with temporal transitions that were adjusted using the most recent scientific evidence.

This library is still a proof-of-concept and it's inteded only to be used for research and experimentation. For more information please read our [preprint](https://arxiv.org/abs/2004.13695):


    Matabuena, M., Meijide-García, C., Rodríguez-Mier, P., & Leborán, V. (2020). 
    COVID-19: Estimating spread in Spain solving an inverse problem with a probabilistic model. 
    arXiv preprint arXiv:2004.13695. https://arxiv.org/abs/2004.13695


This model's main goal is to estimate the levels of infections (or the seroprevalence) of the population, using only data from the registered deaths caused by COVID-19. Although the model can be used to make future predictions (evolution of infections, fatalities, etc.), that's not the primary purpose of the model. Given the uncertainty about essential events that alter the course and dynamics of the spread (for example, the use of masks, lockdowns, social distance, etc.), it is tough to make accurate predictions, so we limit ourselves to use the model to reveal more information about what happened before (backcasting).

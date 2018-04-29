# Easy21 Implementation

This is an implementation of the Easy21 assignment of David Silver's Reinforcement Learning Course at UCL. The assignment can be found [here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf).

## Monte-Carlo Control

`python3 mc.py`

10 Million Episodes of the game have been evaluated, to obtain the following Value function:  
![](mc-value-function)

## TD Learning

`python3 td.py`

Mean Squared Error of the state-action function of the Monte-Carlo experiment with different Lambdas. For each lambda, 10 000 Episodes have been evaluated.  
![](td-mse-lambda)

Mean Squared Error evolution with different Lambdas.  
![](td-mse-episode-lambda)

## Linear Function Approximation

`python3 lfa.py`

The lookup table of the previous experiment is replaced with a linear function approximation. The logic for the feature vector can be found in the assignment.  

![](lfa-mse-lambda)  

![](lfa-mse-episode-lambda)

[mc-value-function]: (figs/mc-value-function.png)
[td-mse-lambda]:
[td-mse-episode-lambda]:
[lfa-mse-lambda]:
[lfa-mse-episode-lambda]:

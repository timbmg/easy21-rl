# Easy21 Implementation

This is an implementation of the Easy21 assignment of David Silver's Reinforcement Learning Course at UCL. The assignment can be found [here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf).

## Monte-Carlo Control

`python3 mc.py`

10 Million Episodes of the game have been evaluated, to obtain the following Value function:  
<img src="https://github.com/timbmg/easy21/blob/master/figs/mc-value-function.png" width="1000">

## TD Learning

`python3 td.py`

Mean Squared Error of the state-action function of the Monte-Carlo experiment with different Lambdas. For each lambda, 10 000 Episodes have been evaluated.  
<img src="https://github.com/timbmg/easy21/blob/master/figs/td-mse-lambda.png" width="1000">

Mean Squared Error evolution with different Lambdas.  
<img src="https://github.com/timbmg/easy21/blob/master/figs/td-mse-episode-lambda.png" width="1000">

## Linear Function Approximation

`python3 lfa.py`

The lookup table of the previous experiment is replaced with a linear function approximation. The logic for the feature vector can be found in the assignment.  

<img src="https://github.com/timbmg/easy21/blob/master/figs/lfa-mse-lambda.png" width="1000">
  
<img src="https://github.com/timbmg/easy21/blob/master/figs/lfa-mse-episode-lambda.png" width="1000">

# Generating set of counterfactuals within distance to optimum

This code used as a starting point an implementation of the paper ["Efficient Search for Diverse
Coherent Explanations"](https://arxiv.org/pdf/1901.04909.pdf) by Chriss Russel.

Original code by Chriss Russell can be found in the [original repository](https://bitbucket.org/ChrisRussell/diverse-coherent-explanations/src/master/).

He used Logistic Regression and focused on generating diverse explanations.
This codebase used his formulation of mixed type inputs (polytopes) and modelling of the
changes of inputs with respect to the original value of the factual. Other
than that, code is my own. The code used was also significantly improved and
clarified as to what the various parts of the code mean.

Instead of diverse coherent explanations, the focus here is on generating a set of
counterfactuals closest to optimal counterfactual.

Examples provided use the adult dataset (included), or MNIST for the mutli
class problem.

The code uses the [Gurobi solver](http://www.gurobi.com/) for the MIP solver, and
[`gurobi-machinelearning`](https://github.com/Gurobi/gurobi-machinelearning)
package for the NN computation.

There is also a custom NN implementation, using the methods presented by M. Fischetti and J. Jo
["Deep neural networks and mixed integer linear optimization"](https://link.springer.com/article/10.1007/s10601-018-9285-6)
That implementation has shown better performance when it comes to speed, but lower
quality of solutions, because it generates duplicate counterfactuals.

## Encoder for data
The encoder explicitly targets the FICO dataset and has made a couple of simple
assumptions as to the form the dataset takes. Each variable is assumed to take a
range of continuous values and a set of discrete values; as simplifying
assumptions we assume that all strictly negative values are the discrete values,
while the continuous values are the non-negative ones.

If you wish to add an entirely discrete variable i.e. without a continious range
included, these variables should be indexed from zero. For example,
in the adult dataset the 'workclass' variable takes the following values.
{0: 'Government', -3: 'Other/Unknown', -2: 'Private', -1: 'Self-Employed'}

If this is not the case for your dataset, the code can be adapted to match
assumptions, but it probably easier to manipulate the data so that it follows
these assumptions -- this manipulation has already been done for the adult
dataset.

## Further own contribution
The encoder was improved from the work of Chriss Russel. The handling of categorical variables is corrected, so now the model works well for categorical, numerical and mixed input features.

## Objective functions
This repository also contains an attempt to create a utility function regarding
the set of counterfactuals. This is a work in progress.

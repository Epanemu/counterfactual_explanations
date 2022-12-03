# Generating set of counterfactuals within distance to optimum

This code used an implementation of the paper ["Efficient Search for Diverse
Coherent Explanations"](https://arxiv.org/pdf/1901.04909.pdf) by Chriss Russel.

Original code can be found in the folder `original/`

Instead of diverse coherent explanations, the focus is on generating a set of
counterfactuals closest to optimal counterfactual.

By default it makes use of the adult dataset (included), or MNIST for the mutli
class problem.

It also uses the [gurobi solver](http://www.gurobi.com/) for the MIP solver, and
[`gurobi-machinelearning`](https://github.com/Gurobi/gurobi-machinelearning)
package for the NN computation. However, a custom
implementation, using the methods presented by M. Fischetti and J. Jo
["Deep neural networks and mixed integer linear optimization"
](https://link.springer.com/article/10.1007/s10601-018-9285-6)

That implementation has shown better performance when it comes to speed, but lower
quality of solutions, because of many duplicate counterfactuals.

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
This is an extended version containing also generation of a set of explanations
closest to the optimum.

Plus the handling of categorical variables has been fixed, so now the model is
correct for categorical, quantitative and mixed input features.

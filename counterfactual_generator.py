#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 30 Sep 21:47:32 CEST 2022

@author: Jiří Němeček

based on the work of Criss Russel
"""
import numpy as np
import gurobipy as gb
from tqdm import tqdm

from collections import namedtuple
Var = namedtuple('Var', ['cont_vars', 'dec_vars', 'orig_val', 'disc_opts'])
CounterFact = namedtuple('CounterFact', ['fact', 'counter_fact', 'orig_class', 'counter_class'])


class CounterfactualGenerator:
    DIFF_TOLERANCE_DECIMALS = 6

    def __init__(self, model, encoder, uniq_bound_M=10e10):
        self.nn_model = model
        self.encoder = encoder
        self.uniq_bound_M = uniq_bound_M

    def __build_structure(self, n_counterfactuals=None, epsilon=None, verbose=False, cf_margin=0):
        """
        build the Core programme of the model that induces
        counterfactuals for the datapoint base_factual
        """
        assert cf_margin >= 0
        assert n_counterfactuals is not None or epsilon is not None, "Must set epsilon as relative distance to optimum or number of closest solutions"

        if not verbose:
            gb.setParam('OutputFlag', 0)
        else:
            gb.setParam('OutputFlag', 1)
            if self.mutli_class:
                print("Model handeled as Multi-class clasifier")
            else:
                print("Model handeled as Binary clasifier")

        self.counterfact_model = gb.Model()

        # --- set the input variable constraints and build x_input ~ the inputs
        x_input = []

        self.vars = np.empty(self.encoder.n_vars, dtype=np.object)
        for i in range(self.encoder.n_vars):
            curr_value = self.base_factual[i]
            curr_context = self.encoder.context[i]
            dec_count = curr_context.median_vals.shape[0]  # number of binary decision variables

            # signs for decision variables [0 for continuous ; 1 for all decision variables, except if one is selected, then -1]
            # decision for continuous does not influence the objective, and since we minimize, the selected is lowering the objective
            sign = np.ones(dec_count)

            if curr_context.scale == 0:  # variable is fully discrete
                disc_index = (curr_context.disc_opts == curr_value).argmax()
                # sign[disc_index] = -1 # Chriss Russel original
                sign[disc_index] = 0  # this seems smarter to me since it wont be a double "cost" for switching from the original. At least for fully discrete variables
                dec_as_input_from = 0  # all decision variables serve as input to the neural network
            else:  # there are some continuous values
                if curr_value < 0:  # current value is < 0 if it is discrete
                    disc_index = (curr_context.disc_opts == curr_value).argmax()
                    sign[disc_index + 1] = -1  # shifted by 1 because at index 0 is dec variable for continuous spectrum
                    # set the value to median value, because when switched to continuous, it should take the median value
                    curr_value = curr_context.median_vals[0]

                sign[0] = 0  # disregard the first decision variable, deciding if it is continuous
                dec_as_input_from = 1  # as input to the neural network are all decision variables, except the one for continuous

            dec_vars = self.counterfact_model.addVars(dec_count, lb=0, ub=1, obj=sign * curr_context.inv_MAD, vtype=gb.GRB.BINARY, name=f"dec{i}")
            dec_vars = np.asarray(dec_vars.values())
            self.counterfact_model.addConstr(dec_vars.sum() == 1)  # (4) in paper, single one must be selected

            if dec_as_input_from == 1:  # need to add continuous variable setup
                cont_ub = np.asarray((curr_value, 1 - curr_value))  # continous values are normalized between 0 and 1
                cont_vars = self.counterfact_model.addVars(2, lb=0, ub=cont_ub, obj=curr_context.inv_MAD[0], name=f"cont{i}")
                cont_vars = np.asarray(cont_vars.values())

                if curr_value == 0:
                    self.counterfact_model.addConstr(cont_vars[1] <= dec_vars[0], name=f"cont_change{i}")  # otherwise we would divide by 0
                elif curr_value == 1:
                    self.counterfact_model.addConstr(cont_vars[0] <= dec_vars[0], name=f"cont_change{i}")  # otherwise we would divide by 0
                else:
                    # disable change of value if discrete decision is made
                    self.counterfact_model.addConstr(
                        cont_vars[0] / cont_ub[0] + cont_vars[1] / cont_ub[1] <= dec_vars[0], name=f"cont_change{i}")

                # add continuous variables to the x_input vector
                x_input.append(curr_value * dec_vars[0] - cont_vars[0] + cont_vars[1])

            self.counterfact_model.update()

            for dvar in dec_vars[dec_as_input_from:]:
                x_input.append(dvar)

            self.vars[i] = Var(cont_vars=cont_vars, dec_vars=dec_vars, orig_val=curr_value, disc_opts=curr_context.disc_opts)

        # setup of the neural network computation within the ILP model
        types, weights, biases = self.nn_model.get_params()
        for t in types:
            if t not in ["linear", "ReLU"]:
                raise f"Not supported type of a layer {t}"

        x_input = np.array(x_input)
        x_prev = self.counterfact_model.addVars(x_input.shape[0], vtype=gb.GRB.CONTINUOUS, name="x_in")
        self.counterfact_model.addConstrs((x_input[i] == x_prev[i] for i in range(x_input.shape[0])), name="set0")
        x_prev = np.array(x_prev.values())
        for i, t in enumerate(types):
            if t != "linear":
                continue
            n_units = biases[i].shape[0]
            if (i + 1 < len(types)) and (types[i + 1] == "ReLU"):
                # according to a paper by Matteo Fischetti, finding tight upper bound and setting it improves the runtime significantly
                pos_next = self.counterfact_model.addVars(n_units, lb=0, vtype=gb.GRB.CONTINUOUS, name=f"pos{i}")
                neg_next = self.counterfact_model.addVars(n_units, lb=0, vtype=gb.GRB.CONTINUOUS, name=f"neg{i}")
                z_indicator = self.counterfact_model.addVars(n_units, vtype=gb.GRB.BINARY, name=f"z{i}")

                # constraints representing layer computation
                self.counterfact_model.addConstrs((
                    weights[i][j].dot(x_prev) + biases[i][j] == pos_next[j] - neg_next[j] for j in range(n_units)),
                    name=f"layer{i}")
                # ReLU indicator constraints
                self.counterfact_model.addConstrs(((z_indicator[j] == 1) >> (pos_next[j] <= 0) for j in range(n_units)), name=f"zpos{i}")
                self.counterfact_model.addConstrs(((z_indicator[j] == 0) >> (neg_next[j] <= 0) for j in range(n_units)), name=f"zneg{i}")
                # reqiure basic uniqueness
                self.counterfact_model.addConstrs((z_indicator[j] <= neg_next[j] * self.uniq_bound_M for j in range(n_units)), name=f"zcheck{i}")

                x_prev = np.array(pos_next.values())  # ReLU makes only the positive values to progress further
            else:
                x_next = self.counterfact_model.addVars(n_units, lb=-np.inf, vtype=gb.GRB.CONTINUOUS, name=f"x{i}")  # unbounded, can be negative
                self.counterfact_model.addConstrs((
                    weights[i][j].dot(x_prev) + biases[i][j] == x_next[j] for j in range(n_units)),
                    name=f"layer{i}")
                x_prev = np.array(x_next.values())

        # cf_margin can ensure strong enough results, but leads to infeasibility if too high
        if x_prev.shape[0] == 1:  # binary classification
            self.counterfact_model.addConstr(x_prev[0] * self.desired_sign >= cf_margin, name="model_result")
        else:
            # set goal according to the mutliclass counterfactual
            if self.goal_class is None:  # any other class
                not_current_class = [i for i in range(x_prev.shape[0]) if i != self.curr_class]
                # for at least one j
                g_indicator = self.counterfact_model.addVars(x_prev.shape[0] - 1, vtype=gb.GRB.BINARY, name="goal")
                self.counterfact_model.addConstrs(
                    ((g_indicator[i] == 1) >> (x_prev[j] - x_prev[self.curr_class] >= cf_margin) for i, j in enumerate(not_current_class)),
                    name="model_result_higher")
                self.counterfact_model.addConstrs(
                    ((g_indicator[i] == 0) >> (x_prev[j] - x_prev[self.curr_class] <= cf_margin) for i, j in enumerate(not_current_class)),
                    name="model_result_lower")
                self.counterfact_model.addConstr(g_indicator.sum() >= 1)  # at least one is higher
            else:  # specific goal class
                not_goal_class = filter(lambda i: i != self.goal_class, range(x_prev.shape[0]))
                self.counterfact_model.addConstrs(
                    (x_prev[self.goal_class] - x_prev[j] >= cf_margin for j in not_goal_class),
                    name="model_result")
            self.goal_layer = x_prev

        self.counterfact_model.setObjective(
            self.counterfact_model.getObjective(), gb.GRB.MINIMIZE)

        # generating in bulk, set apropriate parameters
        if n_counterfactuals is not None:
            self.counterfact_model.setParam("PoolSolutions", n_counterfactuals)  # if epsilon is set, this becomes an upper bound
        if epsilon is not None:
            self.counterfact_model.setParam("PoolGap", epsilon)
        self.counterfact_model.setParam("PoolSearchMode", 2)  # search for closest local optima

        # perform optimization
        self.counterfact_model.optimize()
        if verbose:
            self.counterfact_model.display()

        if self.counterfact_model.status == gb.GRB.INFEASIBLE:
            print("INFEASIBLE MODEL, see file iis_model.ilp for Irreducible Inconsistent Subset (IIS)")
            self.counterfact_model.computeIIS()
            self.counterfact_model.write("iis_model.ilp")

    def __set_factual(self, factual, goal_class=None):
        self.expanded_factual = self.encoder.encode_datapoint(factual)
        prediction = self.nn_model.predict(self.expanded_factual)
        if prediction.shape[0] > 1:
            # model gives mutli class decision
            self.curr_class = np.argmax(prediction, axis=0)
            self.goal_class = goal_class  # None here means that any other class but the current is sought
            self.mutli_class = True
        else:
            # result is binary decision
            self.fact_sign = np.sign(prediction)
            self.desired_sign = -1 * self.fact_sign
            self.mutli_class = False

        self.base_factual = factual.copy().astype(np.float)
        for i in range(factual.size):
            if factual[i] > 0:
                self.base_factual[i] /= self.encoder.context[i].scale  # normalize continuous values

    # ------------------- Helper functions -------------------

    def __recover_val(self, variable):
        dec_values = np.asarray(list(map(lambda x: x.Xn, variable.dec_vars)))
        selected_i = np.argmax(dec_values)

        if dec_values.shape[0] == variable.disc_opts.shape[0]:
            # all decisions are for discrete values, it is a fully discrete variable
            return variable.disc_opts[selected_i]

        cont_vars = variable.cont_vars
        if selected_i == 0:  # return continuous
            return variable.orig_val - cont_vars[0].Xn + cont_vars[1].Xn

        # this should hold true
        assert (np.abs(cont_vars[0].Xn + cont_vars[1].Xn) < 10e-6)

        return variable.disc_opts[selected_i - 1]

    def __recover_all_vals(self):
        return np.asarray(list(map(self.__recover_val, self.vars)))

    # --------------- Counterfactual Generetator functions ----------------------

    def explain_set(self, entries, epsilon=None, n_counterfactuals=None, verbose=False, cf_margin=0, filter_duplicates=True):
        """
        Does not work well if you want to textualize the counterfactuals.
        """
        assert epsilon is not None or n_counterfactuals is not None
        out = []
        for entry in tqdm(entries):
            if epsilon is not None:
                out.append(self.generate_close_counterfactuals(entry, epsilon, verbose=verbose, cf_margin=cf_margin, n_limit=n_counterfactuals, filter_duplicates=filter_duplicates))
            else:
                out.append(self.generate_n_counterfactuals(entry, n_counterfactuals, verbose=verbose, cf_margin=cf_margin, filter_duplicates=filter_duplicates))
        return out

    def __get_counterfactuals(self, filter_duplicates):
        values = []
        classes = []
        for i in range(self.counterfact_model.SolCount):
            self.counterfact_model.setParam("SolutionNumber", i)
            if self.mutli_class:
                cf_class = np.argmax([x.Xn for x in self.goal_layer])
                if self.goal_class is not None:
                    assert cf_class == self.goal_class
            else:
                cf_class = int(self.desired_sign >= 0)
            values.append(self.__recover_all_vals())
            classes.append(cf_class)
        orig_class = self.curr_class if self.mutli_class else int(self.fact_sign >= 0)
        if filter_duplicates:
            rounded = np.around(values, self.DIFF_TOLERANCE_DECIMALS)
            _, indices = np.unique(rounded, return_index=True, axis=0)
            # print(values[:-1] - values[1:])
            values = np.array(values)[indices]
            classes = np.array(classes)[indices]
        results = [
            CounterFact(fact=self.base_factual, counter_fact=cf, orig_class=orig_class, counter_class=cf_c)
            for cf, cf_c in zip(values, classes)
        ]
        return results

    def generate_n_counterfactuals(self, datapoint, n_counterfactuals, verbose=False, cf_margin=0, goal_class=None, filter_duplicates=True):
        assert n_counterfactuals > 0
        self.__set_factual(datapoint, goal_class=goal_class)
        self.__build_structure(n_counterfactuals=n_counterfactuals, verbose=verbose, cf_margin=cf_margin)
        return self.__get_counterfactuals(filter_duplicates)

    def generate_close_counterfactuals(self, datapoint, epsilon, verbose=False, cf_margin=0, n_limit=None, goal_class=None, filter_duplicates=True):
        """
        Can set maximum number of generated counterfactuals with n_limit
        Note that by default, gurobi generates only up to 10 solutions.
        """
        assert epsilon > 0
        self.__set_factual(datapoint, goal_class=goal_class)
        self.__build_structure(epsilon=epsilon, n_counterfactuals=n_limit, verbose=verbose, cf_margin=cf_margin)
        return self.__get_counterfactuals(filter_duplicates)

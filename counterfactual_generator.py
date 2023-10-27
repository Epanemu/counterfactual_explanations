#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 30 Sep 21:47:32 CEST 2022

@author: Jiří Němeček

based on the work of Criss Russel
"""
import numpy as np
import gurobipy as gb
import gurobi_ml
from tqdm import tqdm

from collections import namedtuple
Var = namedtuple('Var', ['cont_vars', 'dec_vars', 'orig_val', 'categ_opts', 'fact_dec_i'])
CounterFact = namedtuple('CounterFact', ['fact', 'counter_fact', 'orig_class', 'counter_class'])


class CounterfactualGenerator:
    DIFF_TOLERANCE_DECIMALS = 6

    def __init__(self, encoder, uniq_bound_M=10e10):
        self.encoder = encoder
        self.uniq_bound_M = uniq_bound_M

    def __build_structure(self, n_counterfactuals=None, epsilon=None, nn_model=None, verbose=False, cf_margin=0):
        """
        build the Core formulation of the model that induces
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

            if curr_context.scale == 0:  # variable is fully categorical
                categ_index = (curr_context.categ_opts == curr_value).argmax()
                # sign[categ_index] = -1 # Chriss Russel original
                sign[categ_index] = 0  # this is better since it wont be a double "cost" for switching from the original. At least for fully categorical variables
                dec_as_input_from = 0  # all decision variables serve as input to the neural network
            else:  # there are some continuous values
                if curr_value < 0:  # current value is < 0 if it is categorical
                    categ_index = (curr_context.categ_opts == curr_value).argmax()
                    categ_index += 1  # shifted by 1 because at index 0 is dec variable for continuous spectrum
                    sign[categ_index] = -1
                    # set the value to median value, because when switched to continuous, it should take the median value
                    curr_value = curr_context.median_vals[0]
                else:
                    categ_index = 0  # continuous value is selected

                sign[0] = 0  # disregard the first decision variable, deciding if it is continuous
                dec_as_input_from = 1  # as input to the neural network are all decision variables, except the one for continuous

            dec_vars = self.counterfact_model.addVars(dec_count, lb=0, ub=1, obj=sign * curr_context.inv_MAD, vtype=gb.GRB.BINARY, name=f"dec{i}")
            if curr_context.categorical_ordered and curr_context.increasing:  # if the categorical values are ordered
                self.counterfact_model.addConstrs((dec_vars[k] == 0 for k in range(dec_as_input_from, categ_index)), name=f"{i}_cannot_get_lower")

            dec_vars = np.asarray(dec_vars.values())
            self.counterfact_model.addConstr(dec_vars.sum() == 1)  # (4) in paper, single one must be selected

            if dec_as_input_from == 1:  # need to add continuous variable setup
                if curr_context.increasing:
                    cont_ub = np.asarray((0, 1 - curr_value))  # the decrease part cannot be present, for example for age
                else:
                    cont_ub = np.asarray((curr_value, 1 - curr_value))  # continous values are normalized between 0 and 1
                cont_vars = self.counterfact_model.addVars(2, lb=0, ub=cont_ub, obj=curr_context.inv_MAD[0], name=f"cont{i}")
                cont_vars = np.asarray(cont_vars.values())

                if curr_value == 0:
                    self.counterfact_model.addConstr(cont_vars[1] <= dec_vars[0], name=f"cont_change{i}")  # otherwise we would divide by 0
                elif curr_value == 1:
                    self.counterfact_model.addConstr(cont_vars[0] <= dec_vars[0], name=f"cont_change{i}")  # otherwise we would divide by 0
                else:
                    # disable change of value if categorical decision is made
                    if curr_context.increasing:  # decreasing part is not present
                        self.counterfact_model.addConstr(cont_vars[1] / cont_ub[1] <= dec_vars[0], name=f"cont_change{i}")
                    else:
                        self.counterfact_model.addConstr(
                            cont_vars[0] / cont_ub[0] + cont_vars[1] / cont_ub[1] <= dec_vars[0], name=f"cont_change{i}")

                # add continuous variables to the x_input vector
                x_input.append(curr_value * dec_vars[0] - cont_vars[0] + cont_vars[1])
            else:
                cont_vars = None

            self.counterfact_model.update()

            for dvar in dec_vars[dec_as_input_from:]:
                x_input.append(dvar)

            self.vars[i] = Var(cont_vars=cont_vars, dec_vars=dec_vars, orig_val=curr_value, categ_opts=curr_context.categ_opts, fact_dec_i=categ_index)

        # causal relationships - if i increases, j must increase as well
        for (i, j) in self.encoder.causal_rels:
            # if categorical - increase is measured in the decision vars
            # TODO implement also when i is not categorical
            if self.encoder.context[i].purely_categ:
                # j can decrease only if i stayed the same or decreased range starts at 0, because it is pure categ. variable
                i_leq_original = sum([self.vars[i].dec_vars[k] for k in range(0, self.vars[i].fact_dec_i + 1)])
                if self.encoder.context[j].purely_categ:
                    j_gt_original = sum([self.vars[j].dec_vars[k] for k in range(self.vars[j].fact_dec_i + 1, len(self.vars[j]))])
                    self.counterfact_model.addConstr(
                        j_gt_original >= 1 - i_leq_original,
                        name=f"{i}->{j}"
                    )
                else:
                    self.counterfact_model.addConstr(
                        self.vars[j].cont_vars[0] <= i_leq_original,
                        name=f"{i}->{j}"
                    )
                    self.counterfact_model.addConstr(
                        self.vars[j].cont_vars[1] >= self.encoder.context[j].epsilon - i_leq_original,
                        name=f"second_{i}->{j}"
                    )

        input_vars = self.counterfact_model.addMVar((len(x_input)), lb=0, ub=1, name="nn_input")
        self.counterfact_model.addConstrs((input_vars[i] == x_input[i] for i in range(len(x_input))), name="nn_input_setup")

        output_vars = self.counterfact_model.addMVar(self.output_shape, lb=-gb.GRB.INFINITY, name="nn_output")

        # setup of the neural network computation within the ILP model
        gurobi_ml.add_predictor_constr(self.counterfact_model, nn_model, input_vars, output_vars)

        # cf_margin can ensure strong enough results, but leads to infeasibility if too high
        if self.output_shape[0] == 1:  # binary classification
            self.counterfact_model.addConstr(output_vars[0] * self.desired_sign >= cf_margin, name="model_result")
        else:
            # set goal according to the mutliclass counterfactual
            if self.goal_class is None:  # any other class
                not_current_class = [i for i in range(self.output_shape[0]) if i != self.curr_class]
                # for at least one j
                g_indicator = self.counterfact_model.addVars(self.output_shape[0] - 1, vtype=gb.GRB.BINARY, name="goal")
                self.counterfact_model.addConstrs(
                    ((g_indicator[i] == 1) >> (output_vars[j] - output_vars[self.curr_class] >= cf_margin) for i, j in enumerate(not_current_class)),
                    name="model_result_higher")
                self.counterfact_model.addConstrs(
                    ((g_indicator[i] == 0) >> (output_vars[j] - output_vars[self.curr_class] <= cf_margin) for i, j in enumerate(not_current_class)),
                    name="model_result_lower")
                self.counterfact_model.addConstr(g_indicator.sum() >= 1)  # at least one is higher
            else:  # specific goal class
                not_goal_class = filter(lambda i: i != self.goal_class, range(self.output_shape[0]))
                self.counterfact_model.addConstrs(
                    (output_vars[self.goal_class] - output_vars[j] >= cf_margin for j in not_goal_class),
                    name="model_result")
            self.goal_layer = output_vars

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
            print("INFEASIBLE MODEL")
            self.counterfact_model.computeIIS()
            self.counterfact_model.write("iis_model.ilp")
            print("see file iis_model.ilp for Irreducible Inconsistent Subset (IIS)")

    def __set_factual(self, factual, model_wrapper, goal_class=None):
        self.expanded_factual = self.encoder.encode_datapoint(factual)
        prediction = model_wrapper.predict(self.expanded_factual)
        self.output_shape = prediction.shape
        if self.output_shape[0] > 1:
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

        if dec_values.shape[0] == variable.categ_opts.shape[0]:
            # all decisions are for categorical values, it is a fully categorical variable
            return variable.categ_opts[selected_i]

        cont_vars = variable.cont_vars
        if selected_i == 0:  # return continuous
            return variable.orig_val - cont_vars[0].Xn + cont_vars[1].Xn

        # this should hold true
        assert (np.abs(cont_vars[0].Xn + cont_vars[1].Xn) < 10e-6)

        return variable.categ_opts[selected_i - 1]

    def __recover_all_vals(self):
        return np.asarray(list(map(self.__recover_val, self.vars)))

    # --------------- Counterfactual Generetator functions ----------------------

    def explain_set(self, entries, model_wrapper, epsilon=None, n_counterfactuals=None, verbose=False, cf_margin=0, filter_duplicates=False):
        """
        Does not work well if you want to textualize the counterfactuals.
        """
        assert epsilon is not None or n_counterfactuals is not None
        out = []
        for entry in tqdm(entries):
            if epsilon is not None:
                out.append(self.generate_close_counterfactuals(entry, model_wrapper, epsilon, verbose=verbose, cf_margin=cf_margin, n_limit=n_counterfactuals, filter_duplicates=filter_duplicates))
            else:
                out.append(self.generate_n_counterfactuals(entry, model_wrapper, n_counterfactuals, verbose=verbose, cf_margin=cf_margin, filter_duplicates=filter_duplicates))
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

    def generate_n_counterfactuals(self, datapoint, model_wrapper, n_counterfactuals, verbose=False, cf_margin=0, goal_class=None, filter_duplicates=False):
        assert n_counterfactuals > 0
        self.__set_factual(datapoint, model_wrapper, goal_class=goal_class)
        self.__build_structure(n_counterfactuals=n_counterfactuals, nn_model=model_wrapper.model, verbose=verbose, cf_margin=cf_margin)
        return self.__get_counterfactuals(filter_duplicates)

    def generate_close_counterfactuals(self, datapoint, model_wrapper, epsilon, verbose=False, cf_margin=0, n_limit=None, goal_class=None, filter_duplicates=False):
        """
        Can set maximum number of generated counterfactuals with n_limit
        Note that by default, gurobi generates only up to 10 solutions.
        """
        assert epsilon > 0
        self.__set_factual(datapoint, model_wrapper, goal_class=goal_class)
        self.__build_structure(epsilon=epsilon, n_counterfactuals=n_limit, nn_model=model_wrapper.model, verbose=verbose, cf_margin=cf_margin)
        return self.__get_counterfactuals(filter_duplicates)

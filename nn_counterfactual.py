#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 30 Sep 21:47:32 CEST 2022

@author: Jiří Němeček

based on the work of Criss Russel
"""
import numpy as np
from types import SimpleNamespace
import gurobipy as gb
# gb.setParam('OutputFlag', 0)


class NNExplanation:
    def __init__(self, model, encoding_size, context):
        self.nn_model = model
        self.encoding_size = encoding_size
        self.context = context

    def build_structure(self, n_explanations=1):
        """build the Core programme of the model that induces
        counterfactuals for the datapoint base_factual"""

        self.counterfact_model = gb.Model()

        # --- set the input variable constraints and build x_input ~ the inputs
        x_input = []

        self.vars = np.empty(self.context.shape[0], dtype=np.object)
        for i in range(self.context.shape[0]):
            curr_value = self.base_factual[i]
            curr_context = self.context[i]
            dec_count = curr_context.median_vals.shape[0] # number of binary decision variables

            # signs for decision variables [0 for continuous ; 1 for all decision variables, except if one is selected, then -1]
            # decision for continuous does not influence the objective, and since we minimize, the selected is lowering the objective
            sign = np.ones(dec_count)
            if curr_value < 0: # current value is < 0 if it is discrete (if all are discrete, current value 0 is also one of the discrete values)
                index = (curr_context.disc_opts == curr_value).argmax()
                sign[index + 1] = -1
                curr_value = curr_context.median_vals[0]
            sign[0] = 0
            cont_ub = np.asarray((curr_value, 1 - curr_value)) # curr_value is normalized between 0 and 1
            cont_vars = self.counterfact_model.addVars(2, lb=0, ub=cont_ub, obj=curr_context.inv_MAD[0], name=f"cont{i}")
            cont_vars = np.asarray(cont_vars.values())
            # there is no minimization of the first decision variable (typically for continuous values), which in case of fully discrete variable
            # makes the choice of option 0 closer and more preferable then options in the negative (-i) form. For these we need to change
            # current option from 1 to 0 and the ith option from 0 to 1. The second change is penalized only for i != 0
            # It is done this way to not penalize the continuous variable more than appropriate, so I would argue that there is a better option of
            # handling this, by introducing another variable and disabling the continuous spectrum
            dec_vars = self.counterfact_model.addVars(dec_count, lb=0, ub=1, obj=sign * curr_context.inv_MAD, vtype=gb.GRB.BINARY, name=f"dec{i}")
            dec_vars = np.asarray(dec_vars.values())
            self.counterfact_model.addConstr(dec_vars.sum() == 1) # (4) in paper, single one must be selected

            if curr_value == 0:
                self.counterfact_model.addConstr(cont_vars[1] <= dec_vars[0], name=f"cont_change{i}") # otherwise we would divide by 0
            elif curr_value == 1:
                self.counterfact_model.addConstr(cont_vars[0] <= dec_vars[0], name=f"cont_change{i}") # otherwise we would divide by 0
            else:
                # disable change ov value if discrete decision is made
                self.counterfact_model.addConstr(
                    cont_vars[0] / cont_ub[0] + cont_vars[1] / cont_ub[1] <= dec_vars[0], name=f"cont_change{i}")

            self.counterfact_model.update()

            # add variables to the x_input vector
            x_input.append(curr_value * dec_vars[0] - cont_vars[0] + cont_vars[1])
            for dvar in dec_vars[1:]:
                x_input.append(dvar)

            self.vars[i] = SimpleNamespace(
                cont_vars=cont_vars, dec_vars=dec_vars, orig_val=curr_value, disc_opts=curr_context.disc_opts)

        # setup the neural network computation within the ILP model
        types, weights, biases = self.nn_model.get_params()
        for t in types:
            if t not in ["linear", "ReLU"]:
                raise f"Not supported type of a layer {t}"

        x_input = np.array(x_input)
        x_prev = self.counterfact_model.addVars(x_input.shape[0], vtype=gb.GRB.CONTINUOUS, name=f"x_in")
        self.counterfact_model.addConstrs((x_input[i] == x_prev[i] for i in range(x_input.shape[0])), name=f"set0")
        x_prev = np.array(x_prev.values())
        for i, t in enumerate(types):
            if t != "linear":
                continue
            n_units = biases[i].shape[0]
            if (i+1 < len(types)) and (types[i+1] == "ReLU"):
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

                x_prev = np.array(pos_next.values()) # ReLU makes only the positive values to progress further
            else:
                x_next = self.counterfact_model.addVars(n_units, vtype=gb.GRB.CONTINUOUS, name=f"x{i}") # unbounded, can be negative
                self.counterfact_model.addConstrs((
                    weights[i][j].dot(x_prev) + biases[i][j] == x_next[j] for j in range(n_units)),
                    name=f"layer{i}")
                x_prev = np.array(x_next.values())

        self.fact_sign = np.sign(self.nn_model.predict(self.expanded_factual))
        self.desired_sign = -1 * self.fact_sign

        # this expects a single output unit for binary clasification
        self.counterfact_model.addConstr(x_prev[0] * self.desired_sign >= 0, name="model_result")
        self.counterfact_model.setObjective(
            self.counterfact_model.getObjective(), gb.GRB.MINIMIZE)

        # generating in bulk, set apropriate parameters
        self.counterfact_model.setParam("PoolSolutions", n_explanations)
        self.counterfact_model.setParam("PoolSearchMode", 2)

        # perform optimization
        self.counterfact_model.optimize()
        self.counterfact_model.display()


    def set_factual(self, factual):
        self.expanded_factual = self.mixed_encode(factual)
        self.base_factual = factual.copy().astype(np.float)
        for i in range(factual.size):
            if factual[i] > 0:
                self.base_factual[i] /= self.context[i].scale # normalize continuous values


    # ------------------- Helper functions -------------------

    def mixed_encode(self, datapoint):
        encoded = np.zeros(self.encoding_size)
        index = 0
        for i in range(self.context.shape[0]): # i as index of data columns
            c = self.context[i]
            if datapoint[i] < 0: # discrete
                val_i = (c.disc_opts == datapoint[i]).argmax()
                encoded[index + val_i + 1] = 1
            else:
                encoded[index] = datapoint[i] / c.scale
            index += c.disc_opts.size + 1
        return encoded

    def recover_val(self, variable):
        dec_values = np.asarray(list(map(lambda x: x.Xn, variable.dec_vars)))
        selected_i = np.argmax(dec_values)
        assert (np.abs(1 - dec_values.sum()) < 10**-4) # check redundant, in case we use this only with the model, this is taken care of by a constraint
        cont_vars = variable.cont_vars
        if selected_i == 0: # return continuous
            return variable.orig_val - cont_vars[0].Xn + cont_vars[1].Xn

        assert (np.abs(cont_vars[0].Xn + cont_vars[1].Xn) < 10**-4) # this should be true since the objective is minimized and both are part of it
        return variable.disc_opts[selected_i - 1]

    def recover_all_val(self):
        return np.asarray(list(map(self.recover_val, self.vars)))


    # ----------------- Textual output helpers -----------------

    # this needs to be set according to your dataset, see example
    string_vals = None

    def format_value(self, value, i):
        val_names = self.string_vals.get(self.context[i].name, self.string_vals)
        str_val = val_names.get(value, '')
        if str_val != "":
            return str_val + f" ({np.round(value).astype(int)})"
        return f"{np.round(value * self.context[i].scale, 2)}"

    def explain(self, values, labels=("'good'", "'bad'")):
        orig_res = labels[int(not self.fact_sign > 0)]
        counterfact = labels[int(not self.desired_sign > 0)]

        mask = np.abs(values - self.base_factual) > 0.001
        explain = (f"You got score {orig_res}.\n" +
                   f"One way you could have got score {counterfact} instead is if:\n")
        changes_str = ""
        for i in range(mask.size):
            if mask[i]:
                changes_str += (f"  {self.context[i].name} had taken value "
                      + f"{self.format_value(values[i], i)} rather than "
                      + f"{self.format_value(self.base_factual[i], i)} and \n")
        explain += changes_str[:-6] # drop " and \n" from the end
        return explain

    def explain_follow(self, values, labels=("'good'", "'bad'")):
        counterfact = labels[int(not self.desired_sign > 0)]

        mask = np.abs(values - self.base_factual) > 0.001
        explain = f"Another way you could have got score {counterfact} instead is if:\n"
        changes_str = ""
        for i in range(mask.size):
            if mask[i]:
                changes_str += (f"  {self.context[i].name} had taken value "
                      + f"{self.format_value(values[i], i)} rather than "
                      + f"{self.format_value(self.base_factual[i], i)} and \n")
        explain += changes_str[:-6] # drop " and \n" from the end
        return explain

    # --------------- Explanations Generetator functions ----------------------

    # def explain_set(self, entries, upto=10):
    #     out = np.empty((entries.shape[0], upto), dtype=np.object)
    #     for i in range(entries.shape[0]):
    #         tmp = np.hstack(self.explain_datapoint(entries[i], upto))
    #         out[i, :tmp.shape[0]] = tmp
    #     return out

    def get_explanations(self, labels):
        explanations = []
        for i in range(self.counterfact_model.SolCount):
            self.counterfact_model.setParam("SolutionNumber", i)
            values = self.recover_all_val()
            if i == 0:
                explanations.append(self.explain(values, labels))
            else:
                explanations.append(self.explain_follow(values, labels))
        return explanations

    def generate_explanations(self, datapoint, n_explanations=100, labels=("'good'", "'bad'")):
        assert n_explanations > 0

        self.set_factual(datapoint)
        self.build_structure(n_explanations)
        return self.get_explanations(labels)

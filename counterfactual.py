#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:24:01 2018

@author: Criss Russel, Jiří Němeček
"""
import numpy as np
from types import SimpleNamespace
import gurobipy as gb
gb.setParam('OutputFlag', 0)


class LinearExplanation:
    def __init__(self, model, encoded, context):
        self.lrmodel = model
        self.encoded_data = encoded
        self.context = context

    def build_structure(self, n_explanations=1):
        """build the Core programme of the model that induces
        counterfactuals for the datapoint base_factual"""
        self.counterfact_model = gb.Model()
        self.fact_direction = np.sign(self.lrmodel.evaluate(self.expanded_factual)) * -1

        intercept, coefficients = self.lrmodel.get_model_params()
        decision = intercept
        model_coeffs = coefficients

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
            cont_vars = self.counterfact_model.addVars(2, lb=0, ub=cont_ub, obj=curr_context.inv_MAD[0])
            cont_vars = np.asarray(cont_vars.values())
            dec_vars = self.counterfact_model.addVars(dec_count, lb=0, ub=1, obj=sign * curr_context.inv_MAD, vtype=gb.GRB.BINARY)
            dec_vars = np.asarray(dec_vars.values())
            self.counterfact_model.addConstr(dec_vars.sum() == 1) # (4) in paper, single one must be selected

            if curr_value == 0:
                self.counterfact_model.addConstr(cont_vars[1] <= dec_vars[0]) # otherwise we would divide by 0
            elif curr_value == 1:
                self.counterfact_model.addConstr(cont_vars[0] <= dec_vars[0]) # otherwise we would divide by 0
            else:
                self.counterfact_model.addConstr(
                    cont_vars[0] / cont_ub[0] + cont_vars[1] / cont_ub[1] <= dec_vars[0]) # disable change ov value if discrete decision is made

            self.counterfact_model.update()

            decision += model_coeffs[0] * (curr_value * dec_vars[0] - cont_vars[0] + cont_vars[1]) # this represents the continuous variable
            if dec_count > 1:
                decision += (model_coeffs[1:dec_count]).dot(dec_vars[1:]) # dot product for computing with the variable that is selected
            model_coeffs = model_coeffs[curr_context.inv_MAD.shape[0]:] # shift coeffs for the next variable
            self.vars[i] = SimpleNamespace(
                cont_vars=cont_vars, dec_vars=dec_vars, orig_val=curr_value, disc_opts=curr_context.disc_opts)

        # D=self.counterfact_model.addVar(-10,10,0,vtype=gb.GRB.CONTINUOUS)
        self.counterfact_model.addConstr(decision * self.fact_direction >= 0) # constraint ensuring to model the other value
        self.counterfact_model.setObjective(
            self.counterfact_model.getObjective(), gb.GRB.MINIMIZE)

        # if generating in bulk, set apropriate parameters
        if n_explanations > 1:
            self.counterfact_model.setParam("PoolSolutions", n_explanations)
            self.counterfact_model.setParam("PoolSearchMode", 2)

        # perform optimization
        self.counterfact_model.optimize()

    def set_factual(self, factual):
        self.expanded_factual = self.mixed_encode(factual)
        self.base_factual = factual.copy().astype(np.float)
        for i in range(factual.size):
            if factual[i] > 0:
                self.base_factual[i] /= self.context[i].scale # normalize continuous values

    # ------------------- Helper functions -------------------

    def mixed_encode(self, datapoint):
        encoded = np.zeros(self.encoded_data.shape[1])
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

    # these need to be set according to your dataset, see example
    string_vals = {-9.0: "No Bureau Record or No Investigation,",
                   -8.0: "No Usable/Valid Accounts Trades or Inquiries,",
                   -7.0: "Condition not Met ,"}

    def format_value(self, value, i):
        val_names = self.string_vals.get(self.context[i].name, self.string_vals)
        str_val = val_names.get(value, '')
        if str_val != "":
            return str_val + f" ({np.round(value).astype(int)})"
        return f"{np.round(value * self.context[i].scale).astype(int)}"

    def explain(self, values, labels=("'good'", "'bad'")):
        orig_res = labels[int(self.fact_direction > 0)]
        counterfact = labels[int(not (self.fact_direction > 0))]

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
        counterfact = labels[int(not (self.fact_direction > 0))]

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

    def modify_model(self, values):
        """Clamp new values and rewind if unsatisfiable"""
        mask = np.abs(values - self.base_factual) > 0.001
        for i in range(mask.size):
            if mask[i]:
                new_const = list()
                if values[i] < 0:
                    v = self.vars[i].dec_vars
                    index = (self.context[i].disc_opts == values[i]).argmax() + 1
                    new_const.append(
                        self.counterfact_model.addConstr(v[index] == 0))
                else:
                    if self.base_factual[i] < 0:
                        index = 0
                        v = self.vars[i].dec_vars
                        new_const.append(
                            self.counterfact_model.addConstr(v[index] == 1))
                    else:
                        v = self.vars[i].cont_vars
                        new_const.append(
                            self.counterfact_model.addConstr(v[0] == 0))
                        new_const.append(
                            self.counterfact_model.addConstr(v[1] == 0))
                self.counterfact_model.update()
                self.counterfact_model.optimize()
                if self.counterfact_model.Status == gb.GRB.INFEASIBLE:
                    for v in new_const:
                        self.counterfact_model.remove(v)
                    self.counterfact_model.update()
                    self.counterfact_model.optimize()


    def give_explanation(self, upto=10, labels=("'good'", "'bad'")):
        """Warning destructive operation. Has different output if rerun without
        rebuilding build_structure.

        However, it can be called repeatedly without rebuilding to generate new
        explanations each time.
        Returns a text string containing upto *upto* different counterfactuals"""
        assert (upto >= 1)
        self.counterfact_model.setParam("SolutionNumber", 0)
        values = self.recover_all_val()

        full_exp = [self.explain(values, labels)]

        prev_values = values
        for _ in range(upto - 1):
            self.modify_model(values)
            values = self.recover_all_val()
            if np.all(values == prev_values): # model gave the same result
                break
            full_exp.append(self.explain_follow(values, labels))
            prev_values = values
        return full_exp

    def explain_datapoint(self, datapoint, upto=10, labels=("'good'", "'bad'")):
        self.set_factual(datapoint)
        self.build_structure()
        text = self.give_explanation(upto, labels)
        return text

    def explain_set(self, entries, upto=10):
        out = np.empty((entries.shape[0], upto), dtype=np.object)
        for i in range(entries.shape[0]):
            tmp = np.hstack(self.explain_datapoint(entries[i], upto))
            out[i, :tmp.shape[0]] = tmp
        return out

    def get_explanations_at_once(self, labels):
        explanations = []
        for i in range(self.counterfact_model.SolCount):
            self.counterfact_model.setParam("SolutionNumber", i)
            values = self.recover_all_val()
            if i == 0:
                explanations.append(self.explain(values, labels))
            else:
                explanations.append(self.explain_follow(values, labels))
        return explanations

    def explain_datapoint_better(self, datapoint, n_explanations=100, labels=("'good'", "'bad'")):
        self.set_factual(datapoint)
        assert n_explanations > 0
        self.build_structure(n_explanations)
        return self.get_explanations_at_once(labels)

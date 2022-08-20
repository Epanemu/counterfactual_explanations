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
    def __init__(self, model, encoded, context, constraints):
        self.lrmodel = model
        self.encoded = encoded
        self.cox = context
        self.constraints = constraints

    def build_structure(self, n_explanations=1):
        """build the Core programme of the model that induces
        counterfactuals for the datapoint test_entry"""
        test_entry = self.short_factual
        self.counterfactual = gb.Model()
        self.direction = np.sign(self.lrmodel.evaluate(self.long_factual)) * -1
        # constant=0
        # Build  polytope
        # Normalise test_entry
        test_entry = np.asarray(test_entry, dtype=np.float).copy()
        # for i in range(self.cox.shape[0]):
        #   if test_entry[i]>=0:
        #       test_entry[i]/=self.cox[i].scale

        # entries

        intercept, coefficients = self.lrmodel.get_model_params()
        decision = intercept
        stack = coefficients

        self.var = np.empty(self.cox.shape[0], dtype=np.object)
        for i in range(self.cox.shape[0]):
            v = test_entry[i]
            c = self.cox[i]
            d_count = c.med.shape[0]
            sign = np.ones(d_count)
            if v < 0:
                index = (c.unique == v).argmax()
                sign[index + 1] = -1
                v = c.med[0]
            sign[0] = 0
            dc = np.asarray((v, 1 - v))
            cvar = self.counterfactual.addVars(2, lb=0, ub=dc, obj=c.MAD[0])
            cvar = np.asarray(cvar.values())
            dvar = self.counterfactual.addVars(d_count, lb=0, ub=1, obj=sign * c.MAD,
                                               vtype=gb.GRB.BINARY)
            dvar = np.asarray(dvar.values())
            self.counterfactual.addConstr(dvar.sum() == 1)
            if v == 0:
                self.counterfactual.addConstr(cvar[1] <= dvar[0])
            elif v == 1:
                self.counterfactual.addConstr(cvar[0] <= dvar[0])
            else:
                self.counterfactual.addConstr(
                    cvar[0] / dc[0] + cvar[1] / dc[1] <= dvar[0])

            self.counterfactual.update()

            decision += stack[0] * (v * dvar[0] - cvar[0] + cvar[1])
            if d_count > 1:
                decision += (stack[1:d_count]).dot(dvar[1:])
            stack = stack[c.MAD.shape[0]:]
            self.var[i] = SimpleNamespace(
                cvar=cvar, dvar=dvar, val=v, unique=c.unique)

        # D=self.counterfactual.addVar(-10,10,0,vtype=gb.GRB.CONTINUOUS)
        self.counterfactual.addConstr(decision * self.direction >= 0)
        self.counterfactual.setObjective(
            self.counterfactual.getObjective(), gb.GRB.MINIMIZE)

        if n_explanations > 1:
            self.counterfactual.setParam("PoolSolutions", n_explanations)
            self.counterfactual.setParam("PoolSearchMode", 2)

        self.counterfactual.optimize()

    def set_factual(self, factual):
        self.long_factual = self.mixed_encode(factual)
        self.short_factual = factual.copy().astype(np.float)
        for i in range(factual.size):
            if factual[i] > 0:
                self.short_factual[i] /= self.cox[i].scale

    # ------------------- Helper functions -------------------

    def mixed_encode(self, test_eg):
        out = np.zeros(self.encoded.shape[1])
        index = 0
        for i in range(self.cox.shape[0]):
            c = self.cox[i]
            if test_eg[i] < 0:
                ind = (c.unique == test_eg[i]).argmax()
                out[ind + index + 1] = 1
            else:
                out[index] = test_eg[i] / c.scale
            index += c.unique.size + 1
        return out

    def recover_val(self, element):
        # print (np.asarray(list(map(lambda x:x.Xn, element.dvar))))
        dval = np.asarray(list(map(lambda x: x.Xn, element.dvar)))
        # print (dval)
        dvar = np.argmax(dval)
        assert (np.abs(1 - dval.sum()) < 10**-4)
        cvar = element.cvar
        if dvar == 0:
            return element.val - cvar[0].Xn + cvar[1].Xn
        assert (np.abs(cvar[0].Xn + cvar[1].Xn) < 10**-4)

        return element.unique[dvar - 1]

    def recover_all_val(self, var):
        return np.asarray(list(map(self.recover_val, var)))

    def recover_stack(self, element):
        dval = np.asarray(list(map(lambda x: x.Xn, element.dvar)))
        out = dval.copy()
        dvar = np.argmax(dval)
        assert (np.abs(1 - dval.sum()) < 10**-4)
        cvar = element.cvar
        if dvar == 0:
            out[0] = element.val - cvar[0].Xn + cvar[1].Xn
        return out

    def recover_all_stack(self, var):
        return np.hstack(list(map(self.recover_stack, var)))

    # ----------------- Textual output helpers -----------------

    # these need to be set according to your dataset, see example
    special_val = {-9.0: ",i.e. No Bureau Record or No Investigation,",
                   -8.0: ",i.e. No Usable/Valid Accounts Trades or Inquiries,",
                   -7.0: ",i.e. Condition not Met ,"}

    def pp(self, x, other, i):
        """pretty printer helper, returns string of negative integer if x<0
            string of floating point value otherwise."""
        spec = self.special_val
        if x < 1e-5:
            return (" %d " % x) + spec.get(self.cox[i].name, spec).get(x, '')
        # out="%1.1f"%(x*self.cox[i].scale)
        # Todo implement rounding up/down  correctly
        if (x - other < 0):
            return "%d" % (np.floor(x * self.cox[i].scale))
        return "%d" % (np.ceil(x * self.cox[i].scale))

    def explain(self, cf, labels=("'good'", "'bad'")):
        test_entry = self.short_factual
        # direction=np.sign(self.evaluate(self.long_factual)) *-1
        actual_score = labels[int(self.direction > 0)]
        cf_score = labels[int(not (self.direction > 0))]
        out = cf
        mask = np.abs(out - test_entry) > 0.001
        explain = ("You got score " + actual_score
                   + ".\n One way you could have got score "
                   + cf_score + " instead is if:\n")
        e = ""
        for i in range(mask.size):
            if mask[i]:
                e += ("  " + self.cox[i].name + " had taken value "
                      + self.pp(out[i], test_entry[i], i) + " rather than "
                      + self.pp(test_entry[i], out[i], i) + ";\n")
        explain += e[:-2]
        return explain

    def explain_follow(self, cf, labels=("'good'", "'bad'")):
        test_entry = self.short_factual
        cf_score = labels[int(not (self.direction > 0))]
        out = cf
        mask = np.abs(out - test_entry) > 0.001
        explain = ("Another way you could have got score "
                   + cf_score + " instead is if:\n")
        e = ""
        for i in range(mask.size):
            if mask[i]:
                e += ("  " + self.cox[i].name + " had taken value "
                      + self.pp(out[i], test_entry[i], i) + " rather than "
                      + self.pp(test_entry[i], out[i], i) + ";\n")
        explain += e[:-2]
        return explain

    def fix_values(self, cf):
        """Clamp new values and rewind if unsatisfiable"""
        out = cf
        test_entry = self.short_factual
        mask = np.abs(out - test_entry) > 0.001
        # print (mask)
        for i in range(mask.size):
            new_const = list()
            if mask[i]:
                # print (i)
                if out[i] < 0:
                    v = self.var[i].dvar
                    index = (self.cox[i].unique == out[i]).argmax() + 1
                    new_const.append(
                        self.counterfactual.addConstr(v[index] == 0))
                else:
                    if test_entry[i] < 0:
                        index = 0
                        v = self.var[i].dvar
                        new_const.append(
                            self.counterfactual.addConstr(v[index] == 1))
                    else:
                        v = self.var[i].cvar
                        new_const.append(
                            self.counterfactual.addConstr(v[0] == 0))
                        new_const.append(
                            self.counterfactual.addConstr(v[1] == 0))
                self.counterfactual.update()
                self.counterfactual.optimize()
                if self.counterfactual.Status == 3:
                    for v in new_const:
                        self.counterfactual.remove(v)
                    self.counterfactual.update()
                    self.counterfactual.optimize()

    def give_explanation(self, upto=10, labels=("'good'", "'bad'")):
        """Warning destructive operation. Has different output if rerun without
        rebuilding build_structure.

        However, it can be called repeatedly without rebuilding to generate new
        explanations each time.
        Returns a text string containing upto *upto* different counterfactuals"""
        assert (upto >= 1)
        self.counterfactual.setParam("SolutionNumber", 0)
        out = self.recover_all_val(self.var)
        full_exp = list()
        full_exp.append(self.explain(out, labels))
        # full_exp+='\n-----\n'
        old_out = out
        for i in range(upto - 1):
            self.fix_values(out)
            out = self.recover_all_val(self.var)
            if np.all(out == old_out):
                break
            full_exp.append(self.explain_follow(out, labels))
            old_out = out
        return full_exp

    def explain_entry(self, entry, upto=10, labels=("'good'", "'bad'")):
        self.set_factual(entry)
        self.build_structure()
        text = self.give_explanation(upto, labels)
        return text

    def explain_set(self, entries, upto=10):
        out = np.empty((entries.shape[0], upto), dtype=np.object)
        for i in range(entries.shape[0]):
            tmp = np.hstack(self.explain_entry(entries[i], upto))
            out[i, :tmp.shape[0]] = tmp
        return out

    def get_explanations_at_once(self, labels):
        explanations = []
        for i in range(self.counterfactual.SolCount):
            self.counterfactual.setParam("SolutionNumber", i)
            all_vals = self.recover_all_val(self.var)
            if i == 0:
                explanations.append(self.explain(all_vals, labels))
            else:
                explanations.append(self.explain_follow(all_vals, labels))
        return explanations

    def explain_entry_better(self, entry, n_explanations=100, labels=("'good'", "'bad'")):
        self.set_factual(entry)
        assert n_explanations > 0
        self.build_structure(n_explanations)
        return self.get_explanations_at_once(labels)

import numpy as np


class Textualizer:
    DIFF_TOLERANCE = 10e-5

    def __init__(self, discrete_names, encoder):
        self.discrete_names = discrete_names
        self.encoder = encoder

    def __format_value(self, value, i):
        val_names = self.discrete_names.get(self.encoder.context[i].name, self.discrete_names)
        str_val = val_names.get(value, '')
        if str_val != "":
            return str_val + f" ({np.round(value).astype(int)})"
        return f"{np.round(value * self.encoder.context[i].scale, 2)}"

    def formulate(self, cf, labels=("negative", "positive")):
        orig_res = labels[cf.orig_class]
        counter_res = labels[cf.counter_class]

        mask = np.abs(cf.counter_fact - cf.fact) > self.DIFF_TOLERANCE
        explanation = (f"You got score {orig_res}.\n"
                       + f"One way you could have got score {counter_res} instead is if:\n")
        changes_str = ""
        for i in range(mask.size):
            if mask[i]:
                changes_str += (f"  {self.encoder.context[i].name} had taken value "
                                + f"{self.__format_value(cf.counter_fact[i], i)} rather than "
                                + f"{self.__format_value(cf.fact[i], i)} and \n")
        explanation += changes_str[:-6]  # drop " and \n" from the end
        return explanation

    def __formulate_follow(self, cf, labels):
        counter_res = labels[cf.counter_class]

        mask = np.abs(cf.counter_fact - cf.fact) > self.DIFF_TOLERANCE
        explanation = f"Another way you could have got score {counter_res} instead is if:\n"
        changes_str = ""
        for i in range(mask.size):
            if mask[i]:
                changes_str += (f"  {self.encoder.context[i].name} had taken value "
                                + f"{self.__format_value(cf.counter_fact[i], i)} rather than "
                                + f"{self.__format_value(cf.fact[i], i)} and \n")
        explanation += changes_str[:-6]  # drop " and \n" from the end
        return explanation

    def formulate_list(self, counterfacts, labels=("negative", "positive")):
        if len(counterfacts) == 0:
            return []
        explanations = [self.formulate(counterfacts[0], labels=labels)]
        for counterfact in counterfacts[1:]:
            explanations.append(self.__formulate_follow(counterfact, labels))
        return explanations

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

    def formulate(self, counterfact, generator, labels=("'good'", "'bad'")):
        if generator.mutli_class:
            orig_res = labels[generator.curr_class]
            label_i = np.argmax([x.Xn for x in generator.goal_layer])
            counter_res = labels[label_i]
            if generator.goal_class is not None:
                assert label_i == generator.goal_class
        else:
            orig_res = labels[int(generator.fact_sign < 0)]
            counter_res = labels[int(generator.desired_sign < 0)]

        mask = np.abs(counterfact - generator.base_factual) > self.DIFF_TOLERANCE
        explanation = (f"You got score {orig_res}.\n" +
                    f"One way you could have got score {counter_res} instead is if:\n")
        changes_str = ""
        for i in range(mask.size):
            if mask[i]:
                changes_str += (f"  {self.encoder.context[i].name} had taken value "
                        + f"{self.__format_value(counterfact[i], i)} rather than "
                        + f"{self.__format_value(generator.base_factual[i], i)} and \n")
        explanation += changes_str[:-6] # drop " and \n" from the end
        return explanation

    def __formulate_follow(self, counterfact, generator, labels):
        if generator.mutli_class:
            label_i = np.argmax([x.Xn for x in generator.goal_layer])
            counter_res = labels[label_i]
            if generator.goal_class is not None:
                assert label_i == generator.goal_class
        else:
            counter_res = labels[int(generator.desired_sign < 0)]

        mask = np.abs(counterfact - generator.base_factual) > self.DIFF_TOLERANCE
        explanation = f"Another way you could have got score {counter_res} instead is if:\n"
        changes_str = ""
        for i in range(mask.size):
            if mask[i]:
                changes_str += (f"  {self.encoder.context[i].name} had taken value "
                        + f"{self.__format_value(counterfact[i], i)} rather than "
                        + f"{self.__format_value(generator.base_factual[i], i)} and \n")
        explanation += changes_str[:-6] # drop " and \n" from the end
        return explanation

    def formulate_list(self, counterfacts, generator, labels=("'good'", "'bad'")):
        explanations = [self.formulate(counterfacts[0], generator, labels=labels)]
        for counterfact in counterfacts[1:]:
            explanations.append(self.__formulate_follow(counterfact, generator, labels))
        return explanations

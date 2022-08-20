import sklearn.linear_model as skl


class LRModel:
    def __init__(self, encoded, target, subset=False):
        """Use optional subset vector to indicate which values correspond to
        the target labels"""
        lr = skl.LogisticRegressionCV(solver='sag')
        if subset is not False:
            self.model = lr.fit(encoded[subset], target)
        else:
            self.model = lr.fit(encoded, target)

    def evaluate(self, datum):
        """Provides a model prediction for a provided datum"""
        # datum=self.recover_all_stack(datum.values)
        datum = datum.reshape(1, -1)
        return self.model.decision_function(datum)

    def get_model_params(self):
        return self.model.intercept_[0], self.model.coef_[0]

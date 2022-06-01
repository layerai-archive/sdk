import layer


class DummyModel(layer.CustomModel):
    def __init__(self):
        super().__init__()

        from sklearn.datasets import load_iris
        from sklearn.svm import SVC

        svc = SVC()
        x, y = load_iris(return_X_y=True)
        svc.fit(x, y)
        self.model = svc

    def predict(self, model_input):
        return self.model.predict(model_input)

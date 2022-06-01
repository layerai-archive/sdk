# type: ignore
import logging
from layer.flavors import CustomModelFlavor

logger = logging.getLogger(__name__)


class TestModelFlavors:

    def test_custom_flavor(self, tmp_path):
        from sklearn.datasets import load_iris
        from sklearn.svm import SVC
        import layer

        from .common.dummy_model import DummyModel

        model = DummyModel()
        flavor = CustomModelFlavor()

        flavor.save_model_to_directory(model, tmp_path)
        loaded_model = flavor.load_model_from_directory(tmp_path).model_object
        assert isinstance(loaded_model, layer.CustomModel)
        assert isinstance(loaded_model.model, SVC)

        x, _ = load_iris(return_X_y=True)
        result = loaded_model.predict(x[:5])
        assert list(result) == [0, 0, 0, 0, 0]

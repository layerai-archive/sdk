# type: ignore
import layer
from layer.flavors import CustomModelFlavor


class TestModelFlavors:
    def test_custom_model_save_load(self, tmp_path):
        import pandas as pd
        from sklearn.svm import SVC

        from layer import CustomModel

        from .common.dummy_model import DummyModel

        model = DummyModel()
        flavor = CustomModelFlavor()

        flavor.save_model_to_directory(model, tmp_path)
        loaded_model = flavor.load_model_from_directory(tmp_path).model_object
        assert isinstance(loaded_model, CustomModel)
        assert isinstance(loaded_model.model, SVC)

        df = pd.DataFrame([[1, 2, 3, 4]])
        result = loaded_model.predict(df)
        assert list(result) is not None

    def test_custom_model_save_fails(self, tmp_path):
        import pandas as pd

        class CustomLocalModel(layer.CustomModel):
            def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
                pass

        model = CustomLocalModel()
        flavor = CustomModelFlavor()

        try:
            flavor.save_model_to_directory(model, tmp_path)
            assert False
        except Exception:
            assert True

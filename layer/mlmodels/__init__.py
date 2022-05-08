from typing import TYPE_CHECKING, Dict, NewType, Union


if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    try:
        MlModelInferableDataset = Union[
            pd.DataFrame,
            np.ndarray,
            Dict[str, np.ndarray],  # type: ignore
        ]
    except ImportError:
        MlModelInferableDataset = Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]  # type: ignore

ModelObject = NewType("ModelObject", object)

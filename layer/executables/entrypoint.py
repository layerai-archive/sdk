import pickle  # nosec import_pickle
from typing import Any, Callable


print("Running Layer Executable")

# load the entrypoint function
with open("function.pkl", "rb") as file:
    entrypoint_function: Callable[..., Any] = pickle.load(file)  # nosec pickle

entrypoint_function()

#! /usr/bin/env python

import pickle  # nosec import_pickle


print("Running Layer Executable")
# load the entrypoint function
with open("function.pkl", "rb") as file:
    entrypoint_function = pickle.load(file)  # nosec pickle
    entrypoint_function()

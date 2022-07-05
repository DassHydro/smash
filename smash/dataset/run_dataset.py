import smash
import matplotlib.pyplot as plt

setup, mesh = smash.load_dataset("Cance")

model = smash.Model(setup, mesh)

model.adjoint_test("gt", inplace=True)


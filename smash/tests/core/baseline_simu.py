from __future__ import annotations

import smash

from smash.solver._mwd_cost import nse, kge
from smash.core._constant import ALGORITHM

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
from h5py import File


def generate_baseline_simu():

    setup, mesh = smash.load_dataset("cance")
    model = smash.Model(setup, mesh)

    with File("baseline_simu.hdf5", "w") as f:

        ### direct run
        instance = model.copy()

        instance.run(inplace=True)

        res = output_cost(instance)

        f.create_dataset("direct_run", data=res)

        del instance

        ### optimize
        for algo in ALGORITHM:

            instance = model.copy()

            if algo=="l-bfgs-b":
                mapping = "distributed" 

            else:
                mapping = "uniform"

            instance.optimize(mapping=mapping, algorithm=algo, options={"maxiter": 2}, inplace=True)

            res = output_cost(instance)

            f.create_dataset(algo, data=res)

            del instance

        ### bayes_estimate
        instance = model.copy()

        br = instance.bayes_estimate(k=np.linspace(-1, 5, 20), n=10, inplace=True, return_br=True, random_state=11)

        f.create_dataset("bayes_estimate_br_cost", data=br.l_curve["cost"])

        res = output_cost(instance)

        f.create_dataset("bayes_estimate", data=res)

        del instance

        ### bayes_optimize
        instance = model.copy()

        br = instance.bayes_optimize(k=np.linspace(-1, 5, 20), n=5, mapping="distributed", algorithm="l-bfgs-b", options={"maxiter": 1}, inplace=True, return_br=True, random_state=11)

        f.create_dataset("bayes_optimize_br_cost", data=br.l_curve["cost"])

        res = output_cost(instance)

        f.create_dataset("bayes_optimize", data=res)

        del instance

        ### ann_optimize with default graph
        instance = model.copy()

        np.random.seed(11)
        net = instance.ann_optimize(epochs=10, inplace=True, return_net=True)

        f.create_dataset("ann_optimize_1_loss", data=net.history["loss_train"])

        res = output_cost(instance)

        f.create_dataset("ann_optimize_1", data=res)

        del instance

        ### ann_optimize with redefined graph
        instance = model.copy()

        problem = instance.get_bound_constraints(states=False)

        nd = instance.input_data.descriptor.shape[-1]
        ncv = problem["num_vars"]
        bounds = problem["bounds"]

        net = smash.Net()

        net.add(layer="dense", options={"input_shape": (nd,), "neurons": 16})
        net.add(layer="activation", options={"name": "relu"})

        net.add(layer="dense", options={"neurons": 8})
        net.add(layer="activation", options={"name": "relu"})

        net.add(layer="dense", options={"neurons": ncv})
        net.add(layer="activation", options={"name": "sigmoid"})

        net.add(layer="scale", options={"bounds": bounds})

        net.compile(optimizer="sgd", learning_rate=0.01, options={'momentum': 0.001}, random_state=11)

        instance.ann_optimize(net=net, epochs=10, inplace=True)

        f.create_dataset("ann_optimize_2_loss", data=net.history["loss_train"])

        res = output_cost(instance)

        f.create_dataset("ann_optimize_2", data=res)

        del instance


def output_cost(instance: Model):

    qo = instance.input_data.qobs
    qs = instance.output.qsim

    ret = []

    for i in range(instance.mesh.code.size):

        ret.append(nse(qo[i], qs[i]))
        ret.append(kge(qo[i], qs[i]))

    return np.array(ret)


if __name__ == '__main__':
    
    generate_baseline_simu()




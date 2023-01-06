from __future__ import annotations

import smash

from smash.core._constant import ALGORITHM, STRUCTURE_PARAMETERS, CSIGN, ESIGN

from core.test_simu import output_cost

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

import numpy as np
from h5py import File


def baseline_simu(model: Model):

    with File("baseline.hdf5", "a") as f:

        ### direct run
        instance = model.copy()

        instance.run(inplace=True)

        res = output_cost(instance)

        f.create_dataset("simu.run", data=res)

        del instance

        ### optimize
        for algo in ALGORITHM:

            instance = model.copy()

            if algo == "l-bfgs-b":
                mapping = "distributed"

            else:
                mapping = "uniform"

            instance.optimize(
                mapping=mapping, algorithm=algo, options={"maxiter": 2}, inplace=True
            )

            res = output_cost(instance)

            f.create_dataset(f"simu.{algo}", data=res)

            del instance

        ### bayes_estimate
        instance = model.copy()

        br = instance.bayes_estimate(
            k=np.linspace(-1, 5, 20),
            n=10,
            inplace=True,
            return_br=True,
            random_state=11,
        )

        f.create_dataset("simu.bayes_estimate_br_cost", data=br.l_curve["cost"])

        res = output_cost(instance)

        f.create_dataset("simu.bayes_estimate", data=res)

        del instance

        ### bayes_optimize
        instance = model.copy()

        br = instance.bayes_optimize(
            k=np.linspace(-1, 5, 20),
            n=5,
            mapping="distributed",
            algorithm="l-bfgs-b",
            options={"maxiter": 1},
            inplace=True,
            return_br=True,
            random_state=11,
        )

        f.create_dataset("simu.bayes_optimize_br_cost", data=br.l_curve["cost"])

        res = output_cost(instance)

        f.create_dataset("simu.bayes_optimize", data=res)

        del instance

        ### ann_optimize with default graph
        instance = model.copy()

        np.random.seed(11)
        net = instance.ann_optimize(epochs=10, inplace=True, return_net=True)

        f.create_dataset("simu.ann_optimize_1_loss", data=net.history["loss_train"])

        res = output_cost(instance)

        f.create_dataset("simu.ann_optimize_1", data=res)

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

        net.compile(
            optimizer="sgd",
            learning_rate=0.01,
            options={"momentum": 0.001},
            random_state=11,
        )

        instance.ann_optimize(net=net, epochs=10, inplace=True)

        f.create_dataset("simu.ann_optimize_2_loss", data=net.history["loss_train"])

        res = output_cost(instance)

        f.create_dataset("simu.ann_optimize_2", data=res)

        del instance


def baseline_signatures(model: Model):

    instance = model.copy()
    instance.run(inplace=True)

    signresult = instance.signatures()
    signsensresult = instance.signatures_sensitivity(n=8, random_state=11)

    with File("baseline.hdf5", "a") as f:

        for typ, sign in zip(["cont", "event"], [CSIGN, ESIGN]):

            for dom in ["obs", "sim"]:

                arr = signresult[typ][dom][sign].to_numpy(dtype=np.float32)

                f.create_dataset(
                    f"signatures.{typ}_{dom}",
                    shape=arr.shape,
                    dtype=arr.dtype,
                    data=arr,
                    compression="gzip",
                    chunks=True,
                )

        for typ, sign in zip(["cont", "event"], [CSIGN, ESIGN]):

            for ord in ["first_si", "total_si"]:

                for param in STRUCTURE_PARAMETERS[instance.setup.structure]:

                    arr = signsensresult[typ][ord][param][sign].to_numpy(
                        dtype=np.float32
                    )

                    f.create_dataset(
                        f"signatures_sens.{typ}_{ord}_{param}",
                        shape=arr.shape,
                        dtype=arr.dtype,
                        data=arr,
                        compression="gzip",
                        chunks=True,
                    )


def baseline_event_seg(model: Model):

    arr = model.event_segmentation().to_numpy()

    arr = arr.astype("S")

    with File("baseline.hdf5", "a") as f:

        f.create_dataset(
            "event_seg",
            shape=arr.shape,
            dtype=arr.dtype,
            data=arr,
            compression="gzip",
            chunks=True,
        )


if __name__ == "__main__":

    setup, mesh = smash.load_dataset("cance")
    model = smash.Model(setup, mesh)

    baseline_simu(model)
    baseline_signatures(model)
    baseline_event_seg(model)

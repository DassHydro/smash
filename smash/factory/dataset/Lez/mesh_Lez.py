import pandas as pd

from smash.factory.dataset.dataset import load_dataset
from smash.factory.mesh.mesh import generate_mesh
from smash.io.mesh import save_mesh

flwdir = load_dataset("flwdir")

gauge_attr = pd.read_csv("gauge_attributes.csv")

mesh = generate_mesh(
    flwdir_path=flwdir,
    x=list(gauge_attr.x),
    y=list(gauge_attr.y),
    area=list(gauge_attr.area * 1e6),
    code=list(gauge_attr.code),
)

save_mesh(mesh, "mesh_Lez.hdf5")

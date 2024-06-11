from smash.factory.dataset.dataset import load_dataset
from smash.factory.mesh.mesh import generate_mesh
from smash.io.mesh import save_mesh

flwdir = load_dataset("flwdir")

bbox_France = (100_000, 1_250_000, 6_050_000, 7_125_000)

mesh = generate_mesh(
    flwdir_path=flwdir,
    bbox=bbox_France,
)

save_mesh(mesh, "mesh_France.hdf5")

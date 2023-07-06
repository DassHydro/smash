from smash.factory.mesh.mesh import generate_mesh
from smash.io.mesh.mesh import save_mesh

from smash.factory.dataset.dataset import load_dataset

flwdir = load_dataset("flwdir")

mesh = generate_mesh(
    flwdir,
    x=[772_330, 772_401, 770_246],
    y=[6_274_127, 6_280_366, 6_284_038],
    area=[169 * 1e6, 143 * 1e6, 113 * 1e6],
    code=["Y3204040", "Y3204030", "Y3204010"],
)

save_mesh(mesh, f"mesh_Lez.hdf5")

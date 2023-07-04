import smash

flwdir = smash.factory.dataset("flwdir")

mesh = smash.factory.mesh(
    flwdir,
    x=[772_330, 772_401, 770_246],
    y=[6_274_127, 6_280_366, 6_284_038],
    area=[169 * 1e6, 143 * 1e6, 113 * 1e6],
    code=["Y3204040", "Y3204030", "Y3204010"],
)

smash.save_mesh(mesh, f"mesh_Lez.hdf5")

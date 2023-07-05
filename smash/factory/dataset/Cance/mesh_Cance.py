import smash

flwdir = smash.factory.load_dataset("flwdir")

mesh = smash.factory.generate_mesh(
    flwdir,
    x=[840_261, 826_553, 828_269],
    y=[6_457_807, 6_467_115, 6_469_198],
    area=[381.7 * 1e6, 107 * 1e6, 25.3 * 1e6],
    code=["V3524010", "V3515010", "V3517010"],
    epsg=2154,
)

smash.save_mesh(mesh, "mesh_Cance.hdf5")

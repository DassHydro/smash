import smash

import time
import numpy as np
import matplotlib.pyplot as plt
# ~ from memory_profiler import profile

# ~ flow_path = "FLOW_fr1km_Leblois_v1_L93.asc"
# ~ flow_path = "30sec_flwdir_SA.tif"

# ~ start_t = time.time()

# ~ mesh = smash.generate_meshing(flow_path, x=-54.15439, y=5.35428, area=76_135 * 1e6, code='MARONI')

# ~ mesh = smash.generate_mesh(
    # ~ flow_path, x=772_363, y=6_274_166, area=168.6 * 1e6, code="Y3204040"
# ~ )

# ~ mesh = smash.generate_meshing(
    # ~ flow_path, x=467_516, y=6_689_246, area=81_314 * 1e6, code="L8000020"
# ~ )

# ~ mesh = smash.generate_meshing(
    # ~ flow_path, x=[467_516, 772_363], y=[6_689_246, 6_274_166], area=[81_314 * 1e6, 168.6 * 1e6], code=["L8000020", "Y3204040"]
# ~ )

# ~ mesh = smash.generate_meshing(
    # ~ flow_path, 
    # ~ x=[770_249, 772_363, 769_922], 
    # ~ y=[6_283_974, 6_274_166, 6_292_568], 
    # ~ area=[113.8 * 1e6, 168.6 * 1e6, 84.6 * 1e6], 
    # ~ code=["Y3204040", "Y3204010", "Y3205010"]
# ~ )

# ~ smash.save_mesh("mesh_Y3204040.hdf5")

meshing_t = time.time()

# ~ print("MESHING", meshing_t - start_t)

mesh = smash.read_mesh("mesh_Y3204040.hdf5")
# ~ mesh = smash.read_mesh("mesh_L8000020.hdf5")

model = smash.Model(configuration="configuration.yaml", mesh=mesh)

# model.run("adj", inplace=True)
model.adjoint_test("gt", inplace=True)
# ~ model_t = time.time()

# ~ print("MODEL", model_t - meshing_t)

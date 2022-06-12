import smash

import matplotlib.pyplot as plt
import time

flow_path = "FLOW_fr1km_Leblois_v1_L93.asc"

start_t = time.time()

# ~ mesh = smash.generate_meshing(
    # ~ flow_path, x=772_363, y=6_274_166, area=168.6 * 1e6, name="Y3204040"
# ~ )
mesh = smash.generate_meshing(
    flow_path, x=467_516, y=6_689_246, area=81_314 * 1e6, name="L8000020"
)

meshing_t = time.time()

print("MESHING", meshing_t - start_t)

model = smash.Model(configuration="configuration.yaml", mesh=mesh)

model_t = time.time()

print("MODEL", model_t - meshing_t)

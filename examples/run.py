import smash
import matplotlib.pyplot as plt

flow_path = "FLOW_fr1km_Leblois_v1_L93.asc"

mesh = smash.generate_meshing(flow_path, x=772_363, y=6_274_166, area=168.6 * 1e6, name='Y3204040')

model = smash.Model(configuration="configuration.yaml", mesh=mesh)

# ~ print(model.mesh.gauge_pos[:,0])
# ~ print(model.mesh.drained_area[26, 12])
# ~ print(model.setup.only_active_cell == True)

plt.imshow(model.mesh.global_active_cell)
plt.show()

# ~ print(model.mesh)

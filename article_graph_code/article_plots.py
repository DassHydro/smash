from utils import *

model_results_dir = './results'
flwdir_path = './dataset/France_flwdir.tif'
info_bv_path = './dataset/Arcmed/catchment_info.csv'
pickle_dir = './results/corr_graph_pickles'

scores_df = scores_per_catchement(model_results_dir)

boxplot_global_scores_nse(scores_df)

boxplot_global_scores_kge(scores_df)

boxplot_spatio_temporal_validation_by_aridity(scores_df)

boxplot_spatio_temporal_validation_by_hourly_discharge(scores_df)

plot_pearson_correlation_difference(model_results_dir)

plot_inferred_parameter_maps(model_results_dir)

cropped_mask, local_meshes =  create_local_mesh_active_cells(info_bv_path, flwdir_path)
stats_df = compute_parameter_stats(model_results_dir, local_meshes)
plot_parameter_stats(stats_df)

linear_cov(pickle_dir=pickle_dir)
desc_map(pickle_dir=pickle_dir)
param_map(pickle_dir=pickle_dir)
import smash
import os
import glob
import h5py

import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import itertools
import rasterio

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import pickle
from sklearn.linear_model import LinearRegression

def nse(observed, simulated):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) for model evaluation.

    Parameters:
        observed (numpy.ndarray): Array of observed data.
        simulated (numpy.ndarray): Array of simulated data.

    Returns:
        float: Nash-Sutcliffe Efficiency (NSE) value.
    """
    simulated = simulated[~np.isnan(observed)]
    observed = observed[~np.isnan(observed)]
    mean_observed = np.nanmean(observed)
    nse = 1 - np.nansum((simulated - observed) ** 2) / np.nansum((observed - mean_observed) ** 2)
    return nse

def kge(observed, simulated):
    """
    Calculate the Kling-Gupta Efficiency (KGE) for model evaluation.

    Parameters:
        observed (numpy.ndarray): Array of observed data.
        simulated (numpy.ndarray): Array of simulated data.

    Returns:
        float: Kling-Gupta Efficiency (KGE) value.
    """
    simulated = simulated[~np.isnan(observed)]
    observed = observed[~np.isnan(observed)]

    mean_obs = np.nanmean(observed)
    mean_sim = np.nanmean(simulated)
    
    std_obs = np.nanstd(observed)
    std_sim = np.nanstd(simulated)
    
    rho = np.corrcoef(np.nan_to_num(observed), np.nan_to_num(simulated))[0, 1]
    
    kge = 1 - np.sqrt((rho - 1) ** 2 + (std_sim / std_obs - 1) ** 2 + (mean_sim / mean_obs - 1) ** 2)
    
    return kge


def compute_flow_errors(model_dir,
                        cal_code_path='dataset/Arcmed/cal.csv',
                        val_code_path='dataset/Arcmed/val.csv'):
    
    scores = []
    
    # read models and extract sm_sim and sm_obs maps for models trained with and without soil moisture
    model_files = glob.glob(os.path.join(model_dir, '*.hdf5'))
    cal_codes = pd.read_csv(cal_code_path)["code"].to_list()
    val_codes = pd.read_csv(val_code_path)["code"].to_list()

    for model_file in model_files:
        base_name = os.path.splitext(os.path.basename(model_file))[0]
        
        if('ann' in base_name):
            model = smash.io.read_model_ddt(model_file)
            qobs = model['response_data']['q']
            qsim = model['response']['q']
            code = model['mesh']['code']
            mapping = 'ann'


        else:
            model = h5py.File(model_file, 'r')['model'] 
            qobs = model['response_data']['q'][:]
            qsim = model['response']['q'][:]
            code = model['mesh']['code'][:]
            codes = [item.decode('utf-8') for item in code]
            code = codes
            mapping = 'distributed'

        for i in range(np.shape(qobs)[0]):

            # Warning some of qobs values are equal to -99
            # clean qobs < 0 
            if(code[i] in cal_codes):
                spatial = 'calibration'
                
            # elif(code[i] in val_codes):
            elif(code[i] in val_codes):
                spatial = 'validation'
            
            else:
                continue

            precentage_of_nans = (sum(qobs[i] < 0)/np.shape(qobs)[1])*100
            qobs[i][qobs[i] < 0] = np.nan
            qsim[i][qobs[i] < 0] = np.nan
            nse_value = nse(qobs[i], qsim[i])
            kge_value = kge(qobs[i], qsim[i])
            id = code[i]
            
            if('without' in base_name):
                sm = False
                new_model_name = base_name[7:]
                sm_metric = 'None'

            else:

                if('spatial_bias' in base_name):
                    sm = True
                    sm_metric = 'sp'
                    new_model_name = base_name[7:]

                else:
                    
                    sm = True
                    sm_metric = 'rmse'
                    new_model_name = base_name[7:]                  
                
            scores.append([new_model_name, mapping, id, precentage_of_nans,  nse_value, kge_value, sm, sm_metric, spatial])


    scores_df = pd.DataFrame(scores, columns=['model', 'mapping', 'code', 'precentage_of_nans', 'NSE', 'KGE', 'used_sm', 'sm_metric', 'spatial'])
    # scores_df = scores_df[scores_df['precentage_of_nans'] != 100]
    
    return scores_df



def scores_per_catchement(model_results_dir):
    calibration_scores_df = compute_flow_errors(model_dir=os.path.join(model_results_dir, 'models_calibrated/Arcmed'))
    validation_scores_df = compute_flow_errors(model_dir=os.path.join(model_results_dir, 'models_validated/Arcmed'))

    calibration_scores_df['period'] = 'calibration'
    validation_scores_df['period'] = 'validation'

    scores_df = pd.concat([calibration_scores_df, validation_scores_df])

    scores_df['new_hue'] = scores_df['mapping']+' with sm as ' + scores_df['sm_metric']


    scores_df.loc[scores_df['new_hue'] == 'ann with sm as None', 'new_hue'] = 'HDA-PR without SM'
    scores_df.loc[scores_df['new_hue'] == 'ann with sm as rmse', 'new_hue'] = 'HDA-PR with SM RMSE'
    scores_df.loc[scores_df['new_hue'] == 'ann with sm as sp', 'new_hue'] = 'HDA-PR with SM SP'
    scores_df.loc[scores_df['new_hue'] == 'distributed with sm as None', 'new_hue'] = 'CDC without SM'
    scores_df.loc[scores_df['new_hue'] == 'distributed with sm as rmse', 'new_hue'] = 'CDC with SM RMSE'
    scores_df.loc[scores_df['new_hue'] == 'distributed with sm as sp', 'new_hue'] = 'CDC with SM SP'

    scores_df.sort_values(by=['new_hue', 'spatial'], inplace=True)

    catchement_info = pd.read_csv(os.path.join(model_results_dir,'catchment_infos.csv'))
    catchement_info.rename(columns={'id': 'code'}, inplace=True)
    scores_df = scores_df.merge(catchement_info, on='code', how='inner')

    return scores_df



def boxplot_spatio_temporal_validation_by_aridity(scores_df):

    subdf = scores_df.query('spatial == "validation" and period == "validation"')

    # Define the bin edges
    bin_edges = [0.6, 1.2, 1.8, 2.4, 3]

    # Define the bin labels
    bin_labels = [f'{bin_edges[i]} - {bin_edges[i+1]}' for i in range(len(bin_edges)-1)]

    # Create the bins
    subdf['Aridity index ranges'] = pd.cut(subdf['aridity_index'], bins=bin_edges, labels=bin_labels)

    fig = plt.figure(figsize=(10, 10))


    # Set the theme and context
    sns.set_theme()
    sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 1.4})

    # Define hatches
    colors = ['#4c72b0', '#dd8452', '#55a868', '#4c72b0', '#dd8452', '#55a868']
    palette = dict(zip(scores_df["new_hue"].unique(), colors))

    # Create the catplot
    g = sns.catplot(x="Aridity index ranges", y="NSE", hue="new_hue",
                    data=subdf, kind="box", height=4, aspect=2, legend=False, palette=palette,
                    width=.85, fliersize=0, linewidth=1.1, notch=False, orient="v")


    # very ugly workaround to set hatches (for now!)
    for i in range(12,24):
        g.axes[0, 0].patches[i].set_hatch('//')


    # Customize titles and limits
    scheme_labels = ['Validation period - Validation catchments']

    i=0
    for i, ax in enumerate(g.axes.flat):
        ax.set_title(scheme_labels[i])
        ax.set_ylim(-0.6, 1.1)
        # Set the hatches and edge color for each box in the plot

    # Adjust the legend
    # Create a custom legend
    handles = []
    hatches = ['', '', '', '//', '//', '//']
    labels = ['CDC with SM RMSE', 'CDC with SM SP', 'CDC without SM', 'HDA-PR with SM RMSE', 'HDA-PR with SM SP', 'HDA-PR without SM']
    for hatch, color, label in zip(hatches, colors, labels):
        patch = mpatches.Patch(facecolor=color, hatch=hatch, label=label, edgecolor='black')
        handles.append(patch)

    plt.legend(handles=handles, title='model', bbox_to_anchor=(1, 1.1))

    # Despine the plot
    sns.despine(trim=True)

    # Show the plot
    plt.savefig('results/figures/spatio-temporal-validation-aridity-index.png', bbox_inches='tight')



def boxplot_global_scores_nse(scores_df):
    # Set the theme and context
    sns.set_theme()
    sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 1.4})

    # Define hatches
    colors = ['#4c72b0', '#dd8452', '#55a868', '#4c72b0', '#dd8452', '#55a868']
    palette = dict(zip(scores_df["new_hue"].unique(), colors))

    # Create the catplot
    g = sns.catplot(x="spatial", y="NSE", hue="new_hue", col="period",
                    data=scores_df, kind="box", height=4, aspect=1, legend=False, palette=palette,
                    width=.85, fliersize=0, linewidth=1.1, notch=True, orient="v")


    # very ugly workaround to set hatches (for now!)

    g.axes[0,0].patches[6].set_hatch('//')
    g.axes[0,0].patches[7].set_hatch('//')
    g.axes[0,0].patches[8].set_hatch('//')
    g.axes[0,0].patches[9].set_hatch('//')
    g.axes[0,0].patches[10].set_hatch('//')
    g.axes[0,0].patches[11].set_hatch('//')
    g.axes[0,1].patches[6].set_hatch('//')
    g.axes[0,1].patches[7].set_hatch('//')
    g.axes[0,1].patches[8].set_hatch('//')
    g.axes[0,1].patches[9].set_hatch('//')
    g.axes[0,1].patches[10].set_hatch('//')
    g.axes[0,1].patches[11].set_hatch('//')

    # Customize titles and limits
    scheme_labels = ['Calibration period', 'Validation period']

    i=0
    for i, ax in enumerate(g.axes.flat):
        ax.set_title(scheme_labels[i])
        ax.set_ylim(-0.6, 1.1)
        # Set the hatches and edge color for each box in the plot

    # Adjust the legend
    # Create a custom legend
    handles = []
    hatches = ['', '', '', '//', '//', '//']
    labels = ['CDC with SM RMSE', 'CDC with SM SP', 'CDC without SM', 'HDA-PR with SM RMSE', 'HDA-PR with SM SP', 'HDA-PR without SM']
    for hatch, color, label in zip(hatches, colors, labels):
        patch = mpatches.Patch(facecolor=color, hatch=hatch, label=label, edgecolor='black')
        handles.append(patch)

    plt.legend(handles=handles, title='model', bbox_to_anchor=(1.1, 1.1))

    # Despine the plot
    # Despine the plot
    sns.despine(trim=True)

    # Show the plot
    plt.savefig('results/figures/global_NSE.png', bbox_inches='tight')


def boxplot_global_scores_kge(scores_df):
    # Set the theme and context
    sns.set_theme()
    sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 1.4})

    # Define hatches
    colors = ['#4c72b0', '#dd8452', '#55a868', '#4c72b0', '#dd8452', '#55a868']
    palette = dict(zip(scores_df["new_hue"].unique(), colors))

    # Create the catplot
    g = sns.catplot(x="spatial", y="KGE", hue="new_hue", col="period",
                    data=scores_df, kind="box", height=4, aspect=1, legend=False, palette=palette,
                    width=.85, fliersize=0, linewidth=1.1, notch=True, orient="v")


    # very ugly workaround to set hatches (for now!)

    g.axes[0,0].patches[6].set_hatch('//')
    g.axes[0,0].patches[7].set_hatch('//')
    g.axes[0,0].patches[8].set_hatch('//')
    g.axes[0,0].patches[9].set_hatch('//')
    g.axes[0,0].patches[10].set_hatch('//')
    g.axes[0,0].patches[11].set_hatch('//')
    g.axes[0,1].patches[6].set_hatch('//')
    g.axes[0,1].patches[7].set_hatch('//')
    g.axes[0,1].patches[8].set_hatch('//')
    g.axes[0,1].patches[9].set_hatch('//')
    g.axes[0,1].patches[10].set_hatch('//')
    g.axes[0,1].patches[11].set_hatch('//')

    # Customize titles and limits
    scheme_labels = ['Calibration period', 'Validation period']

    i=0
    for i, ax in enumerate(g.axes.flat):
        ax.set_title(scheme_labels[i])
        ax.set_ylim(-1.1, 1.1)
        # Set the hatches and edge color for each box in the plot

    # Adjust the legend
    # Create a custom legend
    handles = []
    hatches = ['', '', '', '//', '//', '//']
    labels = ['CDC with SM RMSE', 'CDC with SM SP', 'CDC without SM', 'HDA-PR with SM RMSE', 'HDA-PR with SM SP', 'HDA-PR without SM']
    for hatch, color, label in zip(hatches, colors, labels):
        patch = mpatches.Patch(facecolor=color, hatch=hatch, label=label, edgecolor='black')
        handles.append(patch)

    plt.legend(handles=handles, title='model', bbox_to_anchor=(1.1, 1.1))

    # Despine the plot
    # Despine the plot
    sns.despine(trim=True)

    # Show the plot
    plt.savefig('results/figures/global_KGE.png', bbox_inches='tight')


def boxplot_spatio_temporal_validation_by_hourly_discharge(scores_df):

    subdf = scores_df.query('spatial == "validation" and period == "validation"')

    # Define the bin edges
    bin_edges = [0, 5, 10, 15, 20, 25]

    # Define the bin labels
    bin_labels = [f'{bin_edges[i]} - {bin_edges[i+1]}' for i in range(len(bin_edges)-1)]

    # Create the bins
    subdf['mean_hourly_discharge ranges'] = pd.cut(subdf['mean_hourly_discharge'], bins=bin_edges, labels=bin_labels)

    fig = plt.figure(figsize=(10, 10))


    # Set the theme and context
    sns.set_theme()
    sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 1.4})

    # Define hatches
    colors = ['#4c72b0', '#dd8452', '#55a868', '#4c72b0', '#dd8452', '#55a868']
    palette = dict(zip(scores_df["new_hue"].unique(), colors))

    # Create the catplot
    g = sns.catplot(x='mean_hourly_discharge ranges', y="NSE", hue="new_hue",
                    data=subdf, kind="box", height=4, aspect=2, legend=False, palette=palette,
                    width=.85, fliersize=0, linewidth=1.1, notch=False, orient="v")


    # very ugly workaround to set hatches (for now!)
    for i in range(15,30):
        g.axes[0, 0].patches[i].set_hatch('//')

    # Customize titles and limits
    scheme_labels = ['Validation period - Validation catchments']

    i=0
    for i, ax in enumerate(g.axes.flat):
        ax.set_title(scheme_labels[i])
        ax.set_ylim(-0.6, 1.1)
        # Set the hatches and edge color for each box in the plot

    # Adjust the legend
    # Create a custom legend
    handles = []
    hatches = ['', '', '', '//', '//', '//']
    labels = ['CDC with SM RMSE', 'CDC with SM SP', 'CDC without SM', 'HDA-PR with SM RMSE', 'HDA-PR with SM SP', 'HDA-PR without SM']
    for hatch, color, label in zip(hatches, colors, labels):
        patch = mpatches.Patch(facecolor=color, hatch=hatch, label=label, edgecolor='black')
        handles.append(patch)

    plt.legend(handles=handles, title='model', bbox_to_anchor=(1, 1.1))

    # Despine the plot
    sns.despine(trim=True)

    # Show the plot
    plt.savefig('results/figures/spatio-temporal-validation-mean-hourly-discharge.png', bbox_inches='tight')


# PCC diff
def compute_pearson_correlation_difference(model_dir, hdf5_model_path):
    
    model_hdf5 = h5py.File(hdf5_model_path, 'r')['model']

    active_cells = model_hdf5['mesh']['active_cell'][:]
    index_matrix = np.full(np.shape(active_cells), -99)
    active_indices = np.argwhere(active_cells == 1)
    index_matrix[active_indices[:, 0], active_indices[:, 1]] = np.arange(active_indices.shape[0])
    corr_dict = {}

    sm_obs = np.transpose(np.load(os.path.join(os.path.join(model_dir, 'soil_moisture_results'), 'sm_obs.npy')), (1,0))

    model_files = glob.glob(os.path.join(os.path.join(model_dir, 'soil_moisture_results'), '*.npy'))


    for model_file in model_files:
        
        base_name = os.path.splitext(os.path.basename(model_file))[0]

        if('obs' not in base_name):
            sm_sim = np.load(os.path.join(os.path.join(model_dir, 'soil_moisture_results'), base_name + '.npy'))


            corr_map = np.zeros(sm_obs.shape[1])


            for i in range(sm_obs.shape[1]):

                corr_map[i] = np.corrcoef(sm_sim[1::3,i], sm_obs[::3,i])[0, 1]  

            
            corr_dict[base_name] = np.where(active_cells==1, corr_map[index_matrix], np.nan)



    return corr_dict


def plot_pearson_correlation_difference(model_dir):

    hdf5_model_path = os.path.join(model_dir, 'models_calibrated/Arcmed/Arcmed_distributed_regio_with_sm_rmse.hdf5')
    error_dict_1 = compute_pearson_correlation_difference(model_dir, hdf5_model_path)

    error_dict_1 = dict(sorted(error_dict_1.items()))

    new_error_dict_distributed = {}
    new_error_dict_ann = {}
    new_error_dict_distributed['Arcmed_distributed_regio_with_sm_rmse'] = error_dict_1['Arcmed_distributed_regio_with_sm_rmse'] - error_dict_1['Arcmed_distributed_regio_without_sm_None']
    new_error_dict_distributed['Arcmed_distributed_regio_with_sm_spatial_bias_insensitive'] = error_dict_1['Arcmed_distributed_regio_with_sm_spatial_bias_insensitive'] - error_dict_1['Arcmed_distributed_regio_without_sm_None']
    new_error_dict_ann['Arcmed_ann_regio_with_sm_rmse'] = error_dict_1['Arcmed_ann_regio_with_sm_rmse'] - error_dict_1['Arcmed_ann_regio_without_sm_None']
    new_error_dict_ann['Arcmed_ann_regio_with_sm_spatial_bias_insensitive'] = error_dict_1['Arcmed_ann_regio_with_sm_spatial_bias_insensitive'] - error_dict_1['Arcmed_ann_regio_without_sm_None']

    # Define the colors for the negative, zero, and positive values
    cmap_neg = plt.cm.get_cmap('Blues_r')
    cmap_pos = plt.cm.get_cmap('Reds')

    # Create a new colormap that maps negative values to blue, zero to white, and positive values to red
    cmap_list = [cmap_neg(i) for i in range(128, 256)] + [(1, 1, 1, 1)] + [cmap_pos(i) for i in range(0, 128)]
    cmap_bwrw = colors.LinearSegmentedColormap.from_list('BlueWhiteRedWhite', cmap_list)
    cmap_bwrw.set_bad(color='gray')

    error_dicts = [new_error_dict_distributed, new_error_dict_ann]
    num_rows = len(error_dicts)
    num_cols = len(new_error_dict_ann)

    # Create a figure with subplots
    fig = plt.figure(figsize=(6*num_cols, 5*num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols+1, width_ratios=[1, 1, 0.05])

    # Titles = ['Pearson correlation coefficient (PCC) difference between observed and modeled soil moisture (SM)']
    Titles = [' ']
    col_titles = [['PCC CDC: With SM RMSE - Without SM', 'PCC CDC: With SM SP - Without SM'],
                  ['PCC HDA-PR: With SM RMSE - Without SM', 'PCC HDA-PR: With SM SP - Without SM']]
    for i, error_dict in enumerate(error_dicts):
        # Calculate min and max values across all images in the current dictionary
        min_val = np.inf
        max_val = -np.inf
        for data in error_dict.values():
            data_min = np.nanmin(data)
            # data_max = np.nanmax(data)
            data_max = -data_min

            # data_min = -0.5
            # data_max = 0.5
            if data_min < min_val:
                min_val = data_min
            if data_max > max_val:
                max_val = data_max

        # Create a normalized colormap
        norm = Normalize(vmin=min_val, vmax=max_val)

        # Plot each map in the current dictionary
        for j, (key, data) in enumerate(error_dict.items()):
            ax = plt.subplot(gs[i, j])
            im = ax.imshow(data, cmap=cmap_bwrw, norm=norm)
            # if(j%2 == 0):
            #     ax.set_title('with soil moisture',fontsize=16)
            # else:
            #     ax.set_title('without soil moisture',fontsize=16)
            # title = key.split('_')
            # title = title[1]+' '+title[3]+' '+title[4]+' '+title[5]
            ax.set_title(col_titles[i][j], fontsize=14)

        # Add a colorbar for the current row
        cax = plt.subplot(gs[i, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('PCC difference', fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        # Get the colorbar labels
        # labels = [item.get_text() for item in cbar.ax.get_yticklabels()]

        # # Change the first and last label
        # labels[0] = '<-0.5'
        # labels[-1] = '>0.5'

        # # Set the new labels
        # cbar.ax.set_yticklabels(labels)

        # Add subtitles
    for i, subtitle in enumerate(Titles):
        ax = plt.subplot(gs[i, :])
        ax.set_title(subtitle, fontsize=12+2, fontweight='bold', y=1.05)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('results/figures/PCC_diff.png', bbox_inches='tight')


def compute_error_maps(model_dir='results'):
    
    ci_dict = {}
    cp_dict = {}
    cft_dict = {}
    kexc_dict = {}
    lr_dict = {}

    model_files = glob.glob(os.path.join(os.path.join(model_dir,'models_calibrated/Arcmed') , '*.hdf5'))


    for model_file in model_files:
        
        base_name = os.path.splitext(os.path.basename(model_file))[0]

        if('ann' in base_name):
            model = smash.io.read_model_ddt(model_file)
            mask = model['mesh']['active_cell'][:]
            ci_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,0])
            cp_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,1])
            cft_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,2])
            kexc_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,3])
            lr_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,4])

        else:
            # model = smash.io.read_model(model_file)
            model = h5py.File(model_file, 'r')['model']
            mask = model['mesh']['active_cell'][:]
            ci_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,0])
            cp_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,1])
            cft_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,2])
            kexc_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,3])
            lr_dict[base_name] = np.where(mask == 0, np.nan, model['rr_parameters']['values'][:,:,4])

        ci_dict = dict(sorted(ci_dict.items()))
        cp_dict = dict(sorted(cp_dict.items()))
        cft_dict = dict(sorted(cft_dict.items()))
        kexc_dict = dict(sorted(kexc_dict.items()))
        lr_dict = dict(sorted(lr_dict.items()))


    return ci_dict, cp_dict, cft_dict, kexc_dict, lr_dict


def plot_inferred_parameter_maps(model_dir='results'):

    ci_dict, cp_dict, cft_dict, kexc_dict, lr_dict = compute_error_maps(model_dir)
    error_dict_1 = dict(itertools.islice(cp_dict.items(), 3))
    error_dict_2 = dict(itertools.islice(cft_dict.items(), 3))
    error_dict_3 = dict(itertools.islice(kexc_dict.items(), 3))
    error_dict_4 = dict(itertools.islice(lr_dict.items(), 3))
    error_dicts = [error_dict_1, error_dict_2, error_dict_3, error_dict_4]
    num_rows = len(error_dicts)
    num_cols = len(error_dict_1)

    # Create a figure with subplots
    # fig = plt.figure(figsize=(6*num_cols, 5*num_rows))
    # gs = gridspec.GridSpec(num_rows, num_cols + 1, width_ratios=[1, 1, 1, 1, 1, 1, 0.05])
    fig = plt.figure(figsize=(6*num_cols, 5*num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols + 1, width_ratios=[1, 1, 1, 0.05])

    Titles = ['Capacity of production reservoir (mm)', 'Capacity of transfer reservoir (mm)', 'Non-conservative water exchange flux (mm/dt)', 'linear routing parameter (min)']
    col_titles = ['HDA-PR with SM RMSE', 'HDA-PR with SM SP', 'HDA-PR without SM']
    for i, error_dict in enumerate(error_dicts):
        # Calculate min and max values across all images in the current dictionary
        min_val = np.inf
        max_val = -np.inf
        for data in error_dict.values():
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            if data_min < min_val:
                min_val = data_min
            if data_max > max_val:
                max_val = data_max

        # Create a normalized colormap
        norm = Normalize(vmin=min_val, vmax=max_val)

        # Plot each map in the current dictionary
        for j, (key, data) in enumerate(error_dict.items()):
            ax = plt.subplot(gs[i, j])
            im = ax.imshow(data, cmap='viridis', norm=norm)
            # if(j%2 == 0):
            #     ax.set_title('with soil moisture',fontsize=16)
            # else:
            #     ax.set_title('without soil moisture',fontsize=16)
            # title = key.split('_')
            # title = title[1]+' '+title[3]+' '+title[4]+' '+title[5]
            ax.set_title(col_titles[j], fontsize=16)

        # Add a colorbar for the current row
        cax = plt.subplot(gs[i, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(Titles[i], fontsize=16)
        cbar.ax.tick_params(labelsize=16)

        # Add subtitles
    for i, subtitle in enumerate(Titles):
        ax = plt.subplot(gs[i, :])
        ax.set_title(subtitle, fontsize=12+2, fontweight='bold', y=1.05)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('results/figures/infered_parameter_maps_HDA-PR.png', bbox_inches='tight')


def get_previous_indices(flowdir, indices):

    drow = [1, 1, 0, -1, -1, -1, 0, 1]
    dcol = [0, -1, -1, -1, 0, 1, 1, 1]


    previous_indices = []
    new_indices = []
    

    for index in indices:
        row, col = index
        for i in range(8):
            row_imd = row + drow[i]
            col_imd = col + dcol[i]
            if row_imd < 0 or row_imd >= flowdir.shape[0] or col_imd < 0 or col_imd >= flowdir.shape[1]:
                continue

            if flowdir[row_imd, col_imd] == (i+1):

                previous_indices.append([row_imd, col_imd])
                new_indices.append([row, col])


    return new_indices, previous_indices


def get_upstream_mask(start_position, flowdir):

    previous_indices = [start_position]
    index_mappings = previous_indices
    
    while len(previous_indices) > 0:
        indices, previous_indices = get_previous_indices(flowdir, previous_indices)

        if(len(previous_indices) > 0):
            
            index_mappings.extend(previous_indices)
        
    del previous_indices

    return index_mappings

def lonlat_to_rowcol(lon, lat, min_x, max_y, cell_width, cell_height):
    row = int((max_y - lat ) / cell_height)
    col = int((lon - min_x ) / cell_width )
    return row, col


def create_local_mesh_active_cells(info_bv_path, flwdir_path):

    with rasterio.open(flwdir_path, "r") as src:
        flwdir = src.read(1)
        min_x, max_y = src.transform * (0, 0)

    info_bv = pd.read_csv(info_bv_path)

    gauge_pos = []
    mask_indices = []
    local_meshes = {}

    for i in range(len(info_bv)):
        row,col = lonlat_to_rowcol(info_bv.iloc[i].x, info_bv.iloc[i].y, min_x, max_y, src.res[0], src.res[1])
        area = info_bv.iloc[i].area
        min_tol = np.finfo(float).max

        for j in range(-1,2):
            for k in range(-1,2):
                row_imd = row + j
                col_imd = col + k

                upstream_mask = get_upstream_mask([row_imd, col_imd], flwdir)

                tol = abs(area - len(upstream_mask))/area
                
                if tol > min_tol:
                    continue

                min_tol = tol

                row_dl = row_imd
                col_dl = col_imd
                mask_dl = upstream_mask
        
        
        gauge_pos.append([row_dl, col_dl])
        mask_indices.extend(mask_dl)

        # Initialize a mask matrix of False values
        local_mask = np.full((flwdir.shape), False)

        # Set the positions in the mask matrix to True
        for position in mask_dl:
            local_mask[tuple(position)] = True    

        local_meshes[info_bv.iloc[i].code] = local_mask


    # Initialize a mask matrix of False values
    mask = np.full((flwdir.shape), False)

    # Set the positions in the mask matrix to True
    for position in mask_indices:
        mask[tuple(position)] = True


    # crop to the bounding box of the catchments
    rows, cols = np.nonzero(mask)


    # Adjust the gauge positions
    min_row, min_col = rows.min(), cols.min()
    max_row, max_col = rows.max()+1, cols.max()+1


    cropped_mask = mask[min_row:max_row, min_col:max_col]

    for key in local_meshes.keys():

        local_mask = local_meshes[key]
        cropped_local_mask = local_mask[min_row:max_row, min_col:max_col]

        local_meshes[key] = cropped_local_mask


    # sort the matrices in descending order as to not hide smaller catchments in plots
    # Assuming dict_matrices is your dictionary of boolean matrices
    sorted_matrices = sorted(local_meshes.items(), key=lambda item: np.sum(item[1]), reverse=True)

    # Convert the sorted list back to a dictionary
    sorted_local_meshes = dict(sorted_matrices)

    return cropped_mask, sorted_local_meshes



def plot_stat_maps_per_catchement(stat_dict, cropped_mask):

    global_mean_map = np.zeros(cropped_mask.shape)

    for key in stat_dict.keys():

        local_values = stat_dict[key]

        # get indices where two matrices superspose
        overlap_indices = np.where(global_mean_map.astype(bool) & local_values.astype(bool))
        global_mean_map = local_values + global_mean_map

        global_mean_map[overlap_indices] = local_values[overlap_indices]

    plt.imshow(global_mean_map)
    plt.colorbar()
    plt.show()



def compute_parameter_stats(model_dir, local_meshes):

    model_files = glob.glob(os.path.join(os.path.join(model_dir,'models_calibrated/Arcmed'), '*.hdf5'))

    stats = []

    for model_file in model_files:
        
        base_name = os.path.splitext(os.path.basename(model_file))[0]
        
        if('ann' in base_name):
            model = h5py.File(model_file, 'r')['model_ddt']

        else:

            model = h5py.File(model_file, 'r')['model']
        
        
        for key in local_meshes.keys():

            mask = local_meshes[key]

            ci_mean = np.nanmean(model['rr_parameters']['values'][:,:,0][mask])
            cp_mean = np.nanmean(model['rr_parameters']['values'][:,:,1][mask])
            cft_mean = np.nanmean(model['rr_parameters']['values'][:,:,2][mask])
            kexc_mean = np.nanmean(model['rr_parameters']['values'][:,:,3][mask])
            lr_mean = np.nanmean(model['rr_parameters']['values'][:,:,4][mask])

            ci_std = np.nanstd(model['rr_parameters']['values'][:,:,0][mask])
            cp_std = np.nanstd(model['rr_parameters']['values'][:,:,1][mask])
            cft_std = np.nanstd(model['rr_parameters']['values'][:,:,2][mask])
            kexc_std = np.nanstd(model['rr_parameters']['values'][:,:,3][mask])
            lr_std = np.nanstd(model['rr_parameters']['values'][:,:,4][mask])


            ci_var = ci_std / ci_mean
            cp_var = cp_std / cp_mean
            cft_var = cft_std / cft_mean
            kexc_var = abs(kexc_std / kexc_mean)
            lr_var = lr_std / lr_mean


            stats.append([base_name, key, ci_mean, cp_mean, cft_mean, kexc_mean, lr_mean, ci_std, cp_std, cft_std, kexc_std, lr_std, ci_var, cp_var, cft_var, kexc_var, lr_var])


    stats_df = pd.DataFrame(stats, columns=['model', 'code', 'ci_mean', 'cp_mean', 'cft_mean', 'kexc_mean', 'lr_mean', 'ci_std', 'cp_std', 'cft_std', 'kexc_std', 'lr_std', 'ci_var', 'cp_var', 'cft_var', 'kexc_var', 'lr_var'])
    stats_df.loc[stats_df['model'] == 'Arcmed_ann_regio_without_sm_None', 'model'] = 'HDA-PR without SM'
    stats_df.loc[stats_df['model'] == 'Arcmed_ann_regio_with_sm_rmse', 'model'] = 'HDA-PR with SM RMSE'
    stats_df.loc[stats_df['model'] == 'Arcmed_ann_regio_with_sm_spatial_bias_insensitive', 'model'] = 'HDA-PR with SM SP'

    stats_df.loc[stats_df['model'] == 'Arcmed_distributed_regio_without_sm_None', 'model'] = 'CDC without SM'
    stats_df.loc[stats_df['model'] == 'Arcmed_distributed_regio_with_sm_rmse', 'model'] = 'CDC with SM RMSE'
    stats_df.loc[stats_df['model'] == 'Arcmed_distributed_regio_with_sm_spatial_bias_insensitive', 'model'] = 'CDC with SM SP'

    return stats_df



def plot_parameter_stats(stats_df):

    plot_columns = ['cp_mean', 'cft_mean', 'kexc_mean', 'lr_mean', 'cp_std', 'cft_std', 'kexc_std', 'lr_std', 'cp_var', 'cft_var', 'kexc_var', 'lr_var']

    # Define a list of colors and hatches
    colors = ['#4c72b0', '#dd8452', '#55a868', '#4c72b0', '#dd8452', '#55a868']
    hatches = ['', '', '', '//', '//', '//']

    # Calculate the number of rows needed for the subplots
    nrows = int(np.ceil(len(plot_columns) / 4))

    # Create the subplots
    fig, axs = plt.subplots(nrows, 4, figsize=(20, 5*nrows))

    # If there's only one row, make axs a 2D array for consistency
    if nrows == 1:
        axs = np.array([axs])

    for i, item in enumerate(plot_columns):    

        # Convert the SeriesGroupBy object to a DataFrame
        df = stats_df.groupby('model')[item].apply(list).apply(pd.Series)
        df.reset_index(inplace=True)    
        df = df.T
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df_numeric = df.apply(pd.to_numeric)
        df_numeric = df_numeric.dropna()

        # Calculate the row and column indices for the subplot
        row = i // 4
        col = i % 4

        column_labels = ['Capacity of production reservoir (mm)', 'Capacity of transfer reservoir (mm)', 'Non-conservative water exchange flux (mm/dt)', 'linear routing parameter (min)']
        row_labels = ['Mean', 'Standard Deviation', 'Variation coefficient']  # Replace with your row labels

        if row == 0:
            axs[row, col].set_title(column_labels[col], fontsize=15)

        # Set the row label for the first column
        if col == 0:
            axs[row, col].text(-0.15, 0.5, row_labels[row], rotation=90, verticalalignment='center', horizontalalignment='right', transform=axs[row, col].transAxes, fontsize=15)


        # Create the violinplot on the subplot
        for j in range(df_numeric.shape[1]):
            data = df_numeric.iloc[:, j]
            violin_parts = axs[row, col].violinplot(data, positions=[j], widths=0.7, showmedians=False)

            # Add quantile lines manually
            quartiles = np.percentile(data, [25, 50, 75])
            axs[row, col].scatter([j], quartiles[1], marker='o', color='white', s=30, zorder=3)
            axs[row, col].vlines([j]*3, quartiles[0], quartiles[2], color='#4a4949', linestyle='-', lw=6)

            for pc in violin_parts['bodies']:
                pc.set_facecolor(colors[j % len(colors)])
                pc.set_edgecolor('#4a4949')
                pc.set_hatch(hatches[j % len(hatches)])
                pc.set_alpha(1)

            for part in ('cbars','cmins','cmaxes'):
                vp = violin_parts[part]
                vp.set_edgecolor('#4a4949')

        axs[row, col].set_xticklabels([])  # Remove x-tick labels
        axs[row, col].tick_params(axis='y', labelsize=14)  # Change '14' to the size you want

        # axs[row, col].set_title('')  # Remove title

    # Create legend with hatches
    legend_patches = [mpatches.Patch(facecolor=colors[i], edgecolor='black', hatch=hatches[i], label=df.columns[i]) for i in range(len(df.columns))]
    plt.legend(handles=legend_patches, loc='upper left', fontsize=15, title='Model', title_fontsize='15')

    # Remove any unused subplots
    for j in range(i+1, nrows*4):
        fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    plt.savefig('results/figures/parameter_stats_per_catchement.png', bbox_inches='tight')



def linear_cov(
    pickle_dir,
    params=["cp", "ct", "kexc", "llr"],
    math_params=[r"$c_p$", r"$c_t$", r"$k_{exc}$", r"$l_{l_r}$"],
    figname="linear_cov",
    figsize=(10, 8.5),
):
    print("</> Plotting linear covariance matrix...")
    methods = ['Arcmed_ann_regio_with_sm_rmse', 'Arcmed_ann_regio_with_sm_spatial_bias_insensitive', 'Arcmed_ann_regio_without_sm_None']
    method_labels = ['With SM RMSE', 'With SM SP', 'Without SM']

    with open(os.path.join(pickle_dir, f"descriptors.pickle"), "rb") as f:
        descriptor = pickle.load(f)

    fig, axes = plt.subplots(
        nrows=len(methods), ncols=1, figsize=figsize, constrained_layout=True
    )
    with open(os.path.join(pickle_dir, f"active_cells.pickle"), "rb") as f:
        active_cells = pickle.load(f)

    for k, method in enumerate(methods):
        with open(
            os.path.join(pickle_dir, f"{method}_parameters.pickle"), "rb"
        ) as f:
            parameters = pickle.load(f)

        cov_mat = np.zeros((len(params), 7))

        for j, dei in enumerate(descriptor):
            dei = dei[active_cells==1]

            for i, par_name in enumerate(params):
                pai = parameters[par_name]
                pai = pai[active_cells==1]

                # create a linear regression model
                lm = LinearRegression()

                # fit the model to the data
                lm.fit(dei.reshape(-1, 1), pai.reshape(-1, 1))

                # calculate the predicted values
                pai_pred = lm.predict(dei.reshape(-1, 1)).reshape(pai.shape)

                # calculate the total sum of squares (TSS)
                TSS = ((pai - np.mean(pai)) ** 2).sum()

                # calculate the residual sum of squares (RSS)
                RSS = ((pai - pai_pred) ** 2).sum()

                # calculate R-squared
                cov_mat[i, j] = 1 - (RSS / TSS)

        ytl = [rf"$d_{num + 1}$" for num in range(7) if k == 0]

        axes[k].yaxis.grid(False)
        axes[k].xaxis.grid(False)

        sns.heatmap(
            cov_mat,
            xticklabels=ytl,
            yticklabels=math_params,
            vmin=0,
            vmax=1,
            square=True,
            cbar=(k == len(methods) - 1),
            cbar_kws=dict(
                use_gridspec=False, location="bottom", shrink=0.55, aspect=11.5
            ),
            ax=axes[k],
            cmap="crest",
        )

        axes[k].tick_params(
            labelright=True,
            labelleft=False,
            labelbottom=False,
            labeltop=True,
            labelrotation=0,
        )

        axes[k].set_ylabel(
            f"\n{method_labels[k]}", fontsize=12, labelpad=10
        )

    plt.savefig('results/figures/desc_corr.png', bbox_inches='tight')



def from_params_to_dict(parameters):
    
    params_dict = {}

    params=["cp", "ct", "kexc", "llr"]

    for i, param in enumerate(params):
        params_dict[param] = parameters[:,:,i+1]

    return params_dict


def desc_map(pickle_dir,
    cmap="terrain",
    figname="desc_map",
    figsize=(15, 4),
):
    print("</> Plotting descriptors map...")

    with open(os.path.join(pickle_dir, "descriptors.pickle"), "rb") as f:
        descriptor = pickle.load(f)

    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=figsize)

    for i, darr in enumerate(descriptor):
        axes[i].set_title(rf"$d_{i + 1}$", fontsize=12)

        axes[i].yaxis.grid(False)
        axes[i].xaxis.grid(False)

        axes[i].set_xticks([])
        axes[i].set_yticks([])

        darr[darr<0] = 0
        im = axes[i].imshow(
            darr,
            cmap=cmap,
            interpolation="bicubic",
            alpha=1.0,
        )

        cbar = fig.colorbar(
            im, ax=axes[i], orientation="horizontal", pad=0.1, aspect=15
        )

        cbar.ax.tick_params(labelsize=8)

    plt.savefig('results/figures/desc.png', bbox_inches='tight')


def param_map(pickle_dir,
    params=["cp", "ct", "kexc", "llr"],
    math_params=[r"$c_p$", r"$c_t$", r"$k_{exc}$", r"$l_{l_r}$"],
    bounds=[(100, 900), (0, 150), (-15, 5), (0, 150)],
    cmap="Spectral",
    figname="param_map",
    figsize=(10, 7),
):
    print("</> Plotting parameters map...")

    gs = gridspec.GridSpec(3+1, 4, height_ratios=[1, 1, 1, 0.05])

    fig = plt.figure(figsize=(10, 7))

    methods = ['Arcmed_ann_regio_with_sm_rmse', 'Arcmed_ann_regio_with_sm_spatial_bias_insensitive', 'Arcmed_ann_regio_without_sm_None']
    method_labels = ['With SM RMSE', 'With SM SP', 'Without SM']
    
    with open(os.path.join(pickle_dir, f"active_cells.pickle"), "rb") as f:
        mask = pickle.load(f)

    with open(os.path.join(pickle_dir, "descriptors.pickle"), "rb") as f:
        descriptor = pickle.load(f)
    darr = descriptor[0]

    axes = [[plt.subplot(gs[i, j]) for j in range(4)] for i in range(3)]

    for i, method in enumerate(methods):
        
        with open(
            os.path.join(pickle_dir, f"{method}_parameters.pickle"), "rb"
        ) as f:
            parameters = pickle.load(f)

        for j, par in enumerate(params):
            if i == 0:
                axes[0][j].set_title(math_params[j], fontsize=14)

            axes[i][j].yaxis.grid(False)
            axes[i][j].xaxis.grid(False)

            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

            value = parameters[par]
            value[darr<0] = value[300,400]
            
            im = axes[i][j].imshow(
                value,
                cmap=cmap,
                vmin=bounds[j][0],
                vmax=bounds[j][1],
                interpolation="bicubic",
                alpha=1.0,
            )

            mu = np.mean(value[mask])
            std = np.std(value[mask])
            xlabel = f"$\mu$={str(round(mu, 1))}, $\sigma$={str(round(std, 1))}"

            axes[i][j].set_xlabel(xlabel, labelpad=5, fontsize=12)

            if j == 0:
                axes[i][j].set_ylabel(
                    f"{method_labels[i]}", labelpad=10, fontsize=12
                )

            # if i == 2:
                    # Add a colorbar to the new axes
            clb = fig.colorbar(im, orientation="horizontal", shrink=0.8, location="bottom", cax=plt.subplot(gs[3, j]))
                    # Set fontsize for colorbar
            clb.ax.tick_params(labelsize=10)

    plt.tight_layout(pad=0)
    plt.savefig('results/figures/param_maps.png', bbox_inches='tight')
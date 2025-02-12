.. _user_guide.post_processing_external_tools.results_visualization_over_large_sample:

=======================================
Results Visualization Over Large Sample 
=======================================

The objective of this tutorial is to explain how to work with large data samples. Model calibration and validation using SMASH is done for 30 catchments across France for the purpose of this tutorial. Uniform and distributed optimizations are performed for each catchment over two periods, p1 (from 1 August 2006 to 31 July 2013) and p2 (from 1 August 2013 to 31 July 2020), both periods are used for calibration and validation.
The calibration and validation model objects are saved in hdf5 fomat. The location and the regime of the catchments are shown in the below figure.

.. image:: ../../_static/catchments.png
    :align: center
    :width: 700

Following diagram represents the directory path of the model objects, results and extra files that will be used in this tutorial. The calibrated and validated model objects are saved in the model folder for each set of calculations (un_p1, un_p2, di_p1 and di_p2), un stands for uniform and di for distributed optimization. The extracted data will be stored in the results folder.

.. image:: ../../_static/directory_flow_chart.png
    :align: center
    :width: 700



The libraries which are needed to be imported are as following:

.. code-block:: python

    import smash
    import numpy as np
    import pandas as pd
    import os 
    import csv
    import multiprocessing as mp
    import glob
    from smash.fcore._mw_mask import mask_upstream_cells
    from __future__ import annotations
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from functools import reduce
    import geopandas as gpd 

Data extraction
---------------
This section represents how to extract data (performance_scores, model parameters and signatures) from several model objects using multiprocessing tool.
As an example we want to read and extract data for the model objects of distributed calibration over period p2.
The first code line below assigns the path to the main folder where model objects are saved and the second line sorts the model object files by glob tool. 
Then two DataFrames are generated, df_output which stores the performance scores(nse and kge), calibrated Parameters (cp, ct, kexc and llr) and Continuous hydrological signatures (Runoff coefficient and Wetness index). The second DataFrame,
df1_output, stores the flood event signatures and then both DataFrames are saved as csv files (result.csv and sig.csv). 

.. code-block:: python

    wdir = "..../Data/"
    model_files = sorted(glob.glob(f"{wdir}gr4/di_p2/calibration/model/modelD_*.hdf5"))

    df_output = pd.DataFrame(
        columns = ['code', 'nse', 'kge', 'cp', 'ct', 'kexc', 'llr', 'WI', 'RC_obs', 'RC_sim'])
    df_output.to_csv(f"{wdir}gr4/di_p2/calibration/results/result.csv", header=True, index=False)

    df1_output = pd.DataFrame(
        columns = ['code', 'EPF', 'ELT', 'ERC', 'EFF'])
    df1_output.to_csv(f"{wdir}gr4/di_p2/calibration/results/sig.csv", header=True, index=False)


The following code lines perform all the required operations in order to extract desired data and save them into the already generated csv files. The first code line reads the model objects 
one by one by multiprocessing tool and then performs all the following operations for each model object.

.. code-block:: python

    def compute_scores(model_file, not_used):
        model = smash.io.read_model(model_file)

        # Performance_scores 
        perf_nse = np.round(smash.evaluation(model, metric="nse")[0], 2)
        perf_kge = np.round(smash.evaluation(model, metric="kge")[0], 2)

        '''
        # Parameters: To get parameters for distributed mapping, the parameters array 
        should be multiplied by the mask in order to have parameters only on active cells 
        and then the mean is calculated for each parameter.
        '''
        
        mask = np.zeros((model.mesh.nrow, model.mesh.ncol), order='F', dtype = 'int32')
        mask_upstream_cells(model.mesh, model.mesh.gauge_pos[0][0] + 1, model.mesh.gauge_pos[0][1] + 1, mask)       
        cp=model.get_rr_parameters("cp")*mask
        ct=model.get_rr_parameters("ct")*mask
        kexc=model.get_rr_parameters("kexc")*mask
        llr=model.get_rr_parameters("llr")*mask

        cp_mean = cp[np.nonzero(cp)].mean()
        ct_mean = ct[np.nonzero(ct)].mean()
        kexc_mean = kexc[np.nonzero(kexc)].mean()
        llr_mean = llr[np.nonzero(llr)].mean()

        # Continuous hydrological signatures (Run_off coefficient [RC] and Wetness Index [WI])
        prcp = model.atmos_data.mean_prcp[0, :]
        pet = model.atmos_data.mean_pet[0, :]
        # Indices with no-data precipitation 
        no_data_prcp_indices = np.where(prcp==-99.0)[0] 
        # Indices with no-data evapotranspiration
        no_data_pet_indices = np.where(pet==-99.0)[0] 
        # Combines indices with no-data precipitation and evapotranspiration
        combined_no_data_indices = np.concatenate((no_data_prcp_indices, no_data_pet_indices)) 
        # Deletes the combined no_data indices for precipitation
        prcp = np.delete(prcp, combined_no_data_indices)
        # Deletes the combined no-data indices for evapotranspiration
        pet = np.delete(pet, combined_no_data_indices) 
        prcp_sum=np.sum(prcp)
        pet_sum=np.sum(pet)
        # Wetness Index
        WI = prcp_sum/pet_sum
        sign_obs = smash.signatures(model, domain="obs")
        sign_sim = smash.signatures(model, domain="sim")
        #Runoff Coefficient
        RC_obs = sign_obs.cont["Crc"].values
        RC_sim = sign_sim.cont["Crc"].values

        # Reading the saved result.csv file and storing the extracted data of each model object in it
        df_output = pd.read_csv(f"{wdir}gr4/di_p2/calibration/results/result.csv", header=0)
        df_out_this_run = pd.DataFrame(
            data={
                'code': [model.mesh.code[0]],
                'nse': [perf_nse[0]],
                'kge': [perf_kge[0]],
                'cp': [cp_mean],
                'ct': [ct_mean],
                'kexc': [kexc_mean],
                'llr': [llr_mean],
                'WI': [WI],
                'RC_obs': [RC_obs[0]],
                'RC_sim': [RC_sim[0]]
            }
        )
        df_output = pd.concat([df_output, df_out_this_run])
        df_output.to_csv(f"{wdir}gr4/di_p2/calibration/results/result.csv", header=True, index=False)

        # Error computation for flood event signatures
        EPF = sign_sim.event['Epf']/(sign_obs.event['Epf']) -1
        ELT = sign_sim.event['Elt']-(sign_obs.event['Elt'])
        ERC = sign_sim.event['Erc']/(sign_obs.event['Erc']) -1
        EFF = sign_sim.event['Eff']/(sign_obs.event['Eff']) -1

        # Reading the saved sig.csv file and storing the extracted data of each model object in it
        df1_output = pd.read_csv(f"{wdir}gr4/di_p2/calibration/results/sig.csv", header=0)
        df1_out_this_run = pd.DataFrame(
            data={
                'code': sign_sim.event['code'],
                'EPF': EPF,
                'ELT': ELT,
                'ERC': ERC,
                'EFF': EFF
            }
        )
        df1_output = pd.concat([df1_output, df1_out_this_run],
        )
        df1_output.to_csv(f"{wdir}gr4/di_p2/calibration/results/sig.csv", header=True, index=False)       

    # Below code defines the multiprocessing tool
    pool = mp.Pool(15)
    pool.starmap(compute_scores,[(mf, 1) for mf in model_files])

By the end of the operations we will have two csv files (result.csv and sig.csv) having all the extracted data stored in them. following figures show how the csv files look like.

The result.csv file:

.. image:: ../../_static/result_csv_file.png
    :align: center
    :width: 800



The sig.csv file:

.. image:: ../../_static/signatures_csv_file.png
    :align: center
    :width: 500






Boxplot of performance scores by class
--------------------------------------

Now we can Visualize the results with the extracted data stored inside the two mentioned csv files (result.csv and sig.csv). The aim of this section is to generate boxplot of performance_scores by catchment class.
In the following code lines, root_dir is the path to the main folder containing all stored files, gauge is the BVs_class.csv file containing two columns (catchment code and corresponding class).

.. code-block:: python
    
    root_dir = "..../Data/"
    gauge = pd.read_csv(f"{root_dir}/extra/BVs_class.csv", usecols=["code", "class"])
    gauge.replace({'M': 'Mediterranean', 'O': 'Oceanic', 'U': 'Uniform'}, inplace=True)
    cls = ["Mediterranean", "Oceanic", "Uniform"]

simu_list defines the directory of csv files (result and sig) for each set of calculations (calibration/validation; uniform/distributed). simu_type, perdiod and metric_name defines the type of simulation, period and the score name which we want to plot.

.. code-block:: python

    simu_type = "cal"
    period = "p1"
    metric_name = "nse"

    simu_list = [
        {"simu_type": "cal", "mapping": "u", "period": "p1", "name": "GR4_U", 
        "path": f"{root_dir}/gr4/un_p1/calibration/results"},
        {"simu_type": "cal", "mapping": "d", "period": "p1", "name": "GR4_D", 
        "path": f"{root_dir}/gr4/di_p1/calibration/results"},

        {"simu_type": "val", "mapping": "u", "period": "p1", "name": "GR4_U", 
        "path": f"{root_dir}/gr4/un_p2/validation/results"},
        {"simu_type": "val", "mapping": "d", "period": "p1", "name": "GR4_D", 
        "path": f"{root_dir}/gr4/di_p2/validation/results"},
    ]

Below lines makes a data frame of desired performance_score along with the corresponding code and class of each station using the results.csv file of each set of simulation and the BVs_class.csv file containing code and class of each catchment.

.. code-block:: python

    dat_list = []
    simu_name = []
    for i, simu in enumerate(simu_list):
        
        if simu["simu_type"] == simu_type and simu["period"] == period:

            simu_name.append(simu["name"])
        
            dat = pd.read_csv(f"{simu['path']}/result.csv")
            
            dat = dat.loc[dat["code"].isin(gauge["code"])]
            
            dat.reset_index(drop=True, inplace=True)
            
            dat = dat[["code", metric_name]]
            
            dat.rename(columns={metric_name: simu["name"]}, inplace=True)
            
            dat_list.append(dat)

    df = reduce(lambda x, y: pd.merge(x, y, on='code'), dat_list)
    df = pd.merge(df, gauge, on="code")     
    df.drop(columns=["code"], inplace=True)
    ncls = [len(df.loc[df["class"] == c]) for c in cls]

    arr_values = []
    median_values = []
    for i, cls_name in enumerate(cls):  
        df_imd = df.loc[df["class"] == cls_name].copy()
        df_imd.drop(columns=["class"], inplace=True)
        df_imd_np = df_imd.to_numpy()
        for j, cl in enumerate(list(df_imd)):
            arr_values.append(df_imd_np[:,j])
            median_values.append(round(np.median(df_imd_np[:,j]), 2))

Once the DataFrame is created, the boxplot will be ploted using following code lines which includes different parts (color, title, x_axis, y_axis and legend, ...).

.. code-block:: python

    fig_width = 10
    fig_height = 8
    positions = [1, 1.7, 3, 3.7, 5, 5.7]
    plt.figure(figsize=(fig_width, fig_height))
    colors = ["#5EB1BF", "#EF7B45", "#5EB1BF", "#EF7B45", "#5EB1BF", "#EF7B45"]
    bplt = plt.boxplot(arr_values, positions=positions, 
    medianprops=dict(color="black", linewidth=1.2, ls="solid", alpha=.8), showmeans=False,
    boxprops=dict(color="#565355", linewidth=1.5), whiskerprops=dict(color="#565355", linewidth=1.5),
    capprops=dict(color="#565355", linewidth=1.5), whis=1.5, flierprops=dict(marker="."),
    patch_artist=True, zorder=2)

    for patch, color in zip(bplt["boxes"], colors):
        patch.set_facecolor(color)

    for i, med in enumerate(median_values):
        x = (positions[i] - (min(positions) - 0.5)) / ((max(positions) + 0.5) - (min(positions) - 0.5))
        annot = plt.annotate(med, xy=(x, 1.020), xycoords="axes fraction", ha="center",
        bbox=dict(boxstyle="round4", alpha=0.9, facecolor="white", edgecolor='black'), fontsize=14)

    plt.grid(ls="--", alpha=.7, zorder=1)
    plt.ylim(0, 1)

    if "_" in metric_name:
        name, tfm = (*metric_name.split("_"), )  
        plt.ylabel(f"${name.upper()}$ - {tfm} tfm", fontsize=20)   
    else:
        plt.ylabel(f"${metric_name.upper()}$", fontsize=20)
        
    if simu_type == "cal":   
        title = f"Calibration ${period}$"   
    else:   
        oth_period = "p1" if period == "p2" else "p2"   
        title = f"Validation ${period}$ (with $\hat{{\\theta}}$ of ${oth_period}$)"
        
    plt.yticks(
        ticks = [-1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], 
        labels = ["-1.6", "1.4", "-1.2-", "1", "-0.8", "-0.6", "-0.4", "-0.2", "0", "0.2", "0.4", 
        "0.6", "0.8", "1"], fontsize=14
        )
    xlabels = [f"{c}\n({ncls[i]})" for i, c in enumerate(cls)]
    plt.xticks(ticks=[1.35, 3.35, 5.35], labels=xlabels, fontsize=16, rotation=0)
    plt.title(f"{title}\n", fontsize=18)
    lgd = [name for name in simu_name]
    plt.legend(bplt['boxes'][0:2], lgd, loc='lower left', fontsize=14)
    plt.savefig(f"{root_dir}/bxplt_by_class_{simu_type}_{period}.png", dpi=300)
    plt.close()

By the end of the operations, the above code lines generates the following boxplot of NSE score considering the class of each catchment.

.. image:: ../../_static/bxplt_by_class_cal_p1.png
    :align: center
    :width: 500



Map_cost plot
-------------

In this section we show how to plot performance scores over France map considering the location of each station. The objective is to plot NSE score of validation for both uniform and distributed mappings.
In below code block, France_shp is the France border shapefile and gauge is the BVs_class.csv file which contains code and class of all catchments.

.. code-block:: python

    root_dir = "..../Data/"
    class_colors = {"Mediterranean": "#ffec6e", "Oceanic": "#fccee6", "Uniform": "#8dd3c7"}
    France_shp = gpd.read_file(f"{root_dir}/extra/France_polygone_L93.shp")
    gauge = pd.read_csv(f"{root_dir}/extra/BVs_class.csv")
    metric_name = "nse"

    simu_list = [
        {"simu_type": "val", "mapping": "u", "period": "p1", "name": "GR4_U", 
        "path": f"{root_dir}/gr4/un_p2/validation/results"},
        {"simu_type": "val", "mapping": "d", "period": "p1", "name": "GR4_D", 
        "path": f"{root_dir}/gr4/di_p2/validation/results"}, 
    ]


First we creat DataFrame for uniform mapping by below code lines, the DataFrame includes NSE score of each catchment, latitude and longitude.

.. code-block:: python

    simu_type1 = "val"
    mapping1 = "u"
    period1 = "p1"

    # Extracts the NSE values (Uniform) for each gauge and makes a dataframe (df1)
    dat_list1 = []
    for i, simu in enumerate(simu_list):
        
        if simu["simu_type"] == simu_type1 and simu["mapping"] == mapping1 and simu["period"] == period1:
            
            simu_name1 = simu["name"]
        
            dat1 = pd.read_csv(f"{simu['path']}/result.csv")
            
            dat1 = dat1.loc[dat1["code"].isin(gauge["code"])]
            
            dat1.reset_index(drop=True, inplace=True)
            
            dat1 = dat1[["code", metric_name]]
            
            dat1.rename(columns={metric_name: simu["name"]}, inplace=True)
            
            dat_list1.append(dat1)
                
    df1 = pd.concat(dat_list1, axis=1)
    print(df1)

    # Reading the full_batch_data.csv file which contains the latitue and longitude of each station,
    # generating two new column having the lat and long coordinates and combining it with 
    # the NSE values (Uniform) already in df1
    dat = pd.read_csv(f"{root_dir}/extra/full_batch_data.csv")

    dat = dat.loc[dat["code"].isin(gauge["code"])]

    dat.reset_index(drop=True, inplace=True)

    dat.replace({"PM": "Mediterranean", "PO": "Oceanic"}, inplace=True)

    dat_shp = gpd.GeoDataFrame(dat, geometry=gpd.points_from_xy(dat.x_inrae_l93, dat.y_inrae_l93))

    dat_shp1 = pd.merge(dat_shp, df1, on="code")

    print(dat_shp1)


Now we creat DataFrame for distributed mapping by below code lines, the DataFrame includes NSE score of each catchment, latitude and longitude.

.. code-block:: python

    simu_type2 = "val"
    mapping2 = "d"
    period2 = "p1"

    # Extracts the NSE values (Distributed) for each gauge and makes a dataframe (df2)
    dat_list2 = []
    for i, simu in enumerate(simu_list):
        
        if simu["simu_type"] == simu_type2 and simu["mapping"] == mapping2 and simu["period"] == period2:
            
            simu_name2 = simu["name"]
        
            dat2 = pd.read_csv(f"{simu['path']}/result.csv")
            
            dat2 = dat2.loc[dat2["code"].isin(gauge["code"])]
            
            dat2.reset_index(drop=True, inplace=True)
            
            dat2 = dat2[["code", metric_name]]
            
            dat2.rename(columns={metric_name: simu["name"]}, inplace=True)
            
            dat_list2.append(dat2)
                
    df2 = pd.concat(dat_list2, axis=1)
    print(df2)

    # Reading the full_batch_data.csv file which contains the latitue and longitude of each station,
    # generating two new column having the lat and long coordinates and combining it with
    # the NSE values (Distributed) already in df2.
    dat = pd.read_csv(f"{root_dir}/extra/full_batch_data.csv")

    dat = dat.loc[dat["code"].isin(gauge["code"])]

    dat.reset_index(drop=True, inplace=True)

    dat.replace({"PM": "Mediterranean", "PO": "Oceanic"}, inplace=True)

    dat_shp = gpd.GeoDataFrame(dat, geometry=gpd.points_from_xy(dat.x_inrae_l93, dat.y_inrae_l93))

    dat_shp2 = pd.merge(dat_shp, df2, on="code")

    print(dat_shp2)

Below code block summarizes how to plot the two generated DataFrames along with the colorbar.

.. code-block:: python
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4,2.2))

    France_shp.plot(ax=axs[0], color='white', edgecolor='black', linewidth=.5)
    dat_shp1.plot(ax=axs[0], column=simu_name1, cmap="Spectral", edgecolor='black', 
        linewidth=.5, legend=False, markersize=4, vmin=0, vmax=1)
    axs[0].set_title('GR4_U', fontsize=5, weight='bold')
    France_shp.plot(ax=axs[1], color='white', edgecolor='black', linewidth=.5)
    dat_shp2.plot(ax=axs[1], column=simu_name2, cmap="Spectral", edgecolor='black', 
        linewidth=.5, legend=False, markersize=4, vmin=0, vmax=1,)
    axs[1].set_title('GR4_D', fontsize=5, weight='bold')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap='Spectral'

    # Following two lines makes an space for the colorbar in the figure
    fig.subplots_adjust(right=0.75)
    sub_ax=plt.axes([0.8, 0.27, 0.02, 0.5])
    cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=sub_ax)
    cbar.set_label("NSE", fontsize=5)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(["< 0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=5)
    fig.suptitle("Validation P1 with $\hat{{\\theta}}$ of p2", fontsize=7.5)
    plt.savefig(f"{root_dir}/map_by_cost_{metric_name}_{simu_type1}_{period1}.png", dpi=300)
    plt.close()

The following figure displays the map_NSE plot of the 30 catchments across France map along with the color bar for validation in period p1.

.. image:: ../../_static/map_by_cost_nse_val_p1.png
    :align: center
    :width: 700




Boxplot of signatures by class
------------------------------

To plot signatures, one more csv file is needed to be read in order to be used in defining the code of each catchment and this file can be the sig.csv file of any of the simulation sets that we want to plot its signatures. The reason behind is that for signatures we have multiple events for each catchment while for scores there is just one for each, so in order to show to number of events in the boxplot we need this extra csv file.

.. code-block:: python

    root_dir = "..../Data/"
    gauge = pd.read_csv(f"{root_dir}/extra/BVs_class.csv", usecols=["code", "class"])
    gauge_event = pd.read_csv(f"{root_dir}/gr4/un_p1/calibration/results/sig.csv")

    simu_type = "cal"
    period = "p1"
    metric_name = "ELT"

    simu_list = [
        {"simu_type": "cal", "mapping": "u", "period": "p1", "name": "GR4_U", 
        "path": f"{root_dir}/gr4/un_p1/calibration/results"},
        {"simu_type": "cal", "mapping": "d", "period": "p1", "name": "GR4_D", 
        "path": f"{root_dir}/gr4/di_p1/calibration/results"},

        {"simu_type": "val", "mapping": "u", "period": "p1", "name": "GR4_U", 
        "path": f"{root_dir}/gr4/un_p2/validation/results"},
        {"simu_type": "val", "mapping": "d", "period": "p1", "name": "GR4_D", 
        "path": f"{root_dir}/gr4/di_p2/validation/results"},
    ]

The rest of code lines remains the same as in Boxplot of performance scores by class section except for creating the DataFrame which is as following.

.. code-block:: python 

    dat_list = []
    simu_name = []

    for i, simu in enumerate(simu_list):
        
        if simu["simu_type"] == simu_type and simu["period"] == period:

            simu_name.append(simu["name"])
        
            dat = pd.read_csv(f"{simu['path']}/sig.csv")
            
            dat = dat.loc[dat["code"].isin(gauge_event["code"])]
            
            dat.reset_index(drop=True, inplace=True)
            
            dat = dat[["code", metric_name]]
            
            dat.rename(columns={metric_name: simu["name"]}, inplace=True)
            
            dat_list.append(dat)

    df = pd.concat(dat_list, axis=1)

    df1 = df.iloc[:,:2]
    df2 = df.iloc[:,2:]

    df1.sort_values(by=['code'], ascending = True, inplace=True)
    df2.sort_values(by=['code'], ascending = True, inplace=True)

    df1 = pd.merge(df1, gauge, on="code")     
    df2 = pd.merge(df2, gauge, on="code") 

    df = pd.concat([df1['GR4_U'], df2['GR4_D'], df2['class']], axis=1)

Below figure show the boxplot of ELT (lag time) for the calibration period p1.

.. image:: ../../_static/bxplt_by_class_ELT_cal_p1.png
    :align: center
    :width: 500




Scatterplot of parameters
-------------------------

In this section we want to plot calibrated parameters for both periods of p1 and p2. Three csv files are needed which are the result.csv files for p1 and p2 which contains calibrated parameters and the BVs_class.csv file for class.

.. code-block:: python

    root_dir = "..../Data/"
    structure_name = "GR4_U"
    STRUCTURE_PARAMETERS = {
        "GR4_U": ["cp", "ct", "kexc", "llr"],
    }

    dat_p1 = pd.read_csv(f"{root_dir}/gr4/un_p1/calibration/results/result.csv")
    dat_p2 = pd.read_csv(f"{root_dir}/gr4/un_p2/calibration/results/result.csv")

    gauge = pd.read_csv(f"{root_dir}/extra/BVs_class.csv", usecols=["code", "class"])
    gauge.replace({'M': 'Mediterranean', 'O': 'Oceanic', 'U': 'Uniform'}, inplace=True)
    cls = ["Mediterranean", "Oceanic", "Uniform"]
    cls_colors = {"Mediterranean": "#ffec6e", "Oceanic": "#fccee6", "Uniform": "#8dd3c7"}

    dat_p1 = pd.merge(dat_p1, gauge, on="code")
    dat_p2 = pd.merge(dat_p2, gauge, on="code")

    f, ax = plt.subplots(2, 2, figsize=(15,10))
    math_parameters = {
        "cp": "$\\overline{c_{p}}$ (mm)", 
        "ct": "$\\overline{c_{t}}$ (mm)", 
        "kexc": "$\\overline{k_{exc}}$ (mm/h)", 
        "llr": "$\\overline{l_{lr}}$ (min)", 
    }

    for i, parameter in enumerate(STRUCTURE_PARAMETERS[structure_name]):       
        row = i // 2
        col = i % 2
        
        for c in cls:
            cls_dat_p1 = dat_p1.loc[dat_p1["class"] == c].copy()
            cls_dat_p2 = dat_p2.loc[dat_p2["class"] == c].copy()
            
            x = cls_dat_p1[parameter ]
            y = cls_dat_p2[parameter ]
        
            ax[row, col].plot(x, y, ls="", marker=".", color=cls_colors[c], ms=10, mec="black", mew=0.5, zorder=2)
            ax[row, col].grid(alpha=.7, ls="--")
            ax[row, col].set_xlabel(math_parameters[parameter] + " $p1$", fontsize=14)
            ax[row, col].set_ylabel(math_parameters[parameter] + " $p2$", fontsize=14)
            

        t_x = dat_p1[parameter]
        t_y = dat_p2[parameter]
        
        t_min = np.minimum(np.min(t_x), np.min(t_y))
        t_max = np.maximum(np.max(t_x), np.max(t_y))
        
        ax[row, col].plot([t_min, t_max], [t_min, t_max], color="black", ls="--", alpha=.8, zorder=1)

    f.legend(cls, loc='upper center')
    plt.savefig(f"{root_dir}/scatter_parameters.png", dpi=300)
    plt.show()

The scatterplot generated by above code lines is shown below.

.. image:: ../../_static/scatter_parameters.png
    :align: center
    :width: 800

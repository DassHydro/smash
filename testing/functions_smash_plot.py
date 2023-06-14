import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
import smash
import h5py
import os
import pandas as pd


import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from functions_smash_time import *


def plot_discharges(model,title="",figname="",columns=[],xlim=[None,None],ylim=[None,None],linewidth=1.5,legend=True,color=["black","grey","blue"],plot_rainfall=True,plot=None):
    
    #manage date here
    #compute date_range
    date_deb=datetime.datetime.fromisoformat(model.setup.start_time)+duration_to_timedelta(int(model.setup.dt))
    date_end=datetime.datetime.fromisoformat(model.setup.end_time)+duration_to_timedelta(int(model.setup.dt))
    date_range=[date_deb,date_end,model.setup.dt]
    
    # ~ plot=plt.subplots()
    if plot is None:
        if plot_rainfall:
            fig, (ax1, ax2) = plt.subplots(2, 1,height_ratios=[1, 4])
            fig.subplots_adjust(hspace=0)
            plot=[fig, ax2, ax1]
        else:
            fig, ax2 = plt.subplots()
            plot=[fig, ax2]
    else:
        if plot_rainfall:
            fig=plot[0]
            ax1=plot[2]
            ax2=plot[1]
        else:
            fig=plot[0]
            ax2=plot[1]
        
    plot=plot_time_vars(model.input_data.qobs,columns=[0],title=title,label="Observations at "+model.mesh.code[0],xlabel="Time step",dt=0,ylabel="Discharge $(m^3/s)$",figname=figname,color=color[0],linestyle="--",linewidth=linewidth,xlim=xlim,ylim=ylim,legend=legend,date_range=date_range,plot=plot)
    
    plot=plot_time_vars(model.output.qsim,columns=[0],title=title,label="Simulation at "+model.mesh.code[0],xlabel="Time step",dt=0,ylabel="Discharge $(m^3/s)$",figname=figname,color=color[1],linestyle="-",linewidth=linewidth,legend=legend,xlim=xlim,ylim=ylim,date_range=date_range,plot=plot)
    
    
    xtics = np.arange(np.datetime64(date_range[0]),np.datetime64(date_range[1]), np.timedelta64(int(date_range[2]), 's'))
    
    if plot_rainfall:
    
        ax1.bar(xtics,model.input_data.mean_prcp[0,:],label="Average rainfall (mm)")
        ax1.invert_yaxis()
        ax1.grid(alpha=.7, ls="--")
        ax1.get_xaxis().set_visible(False)
        ax1.set_ylim(bottom=1.2*max(model.input_data.mean_prcp[0,:]))
        ax1.set_ylabel('Average rainfall (mm)')
        
        if legend:
            ax1.legend(loc='upper right')
        else:
            ax1.legend(loc='upper right').set_visible(False)
        
        plot=[fig, ax2, ax1]
    else:
        plot=[fig, ax2]
    
    return plot





def plot_discharges_hdf5(hdf5,title="",figname="",columns=[],xlim=[None,None],ylim=[None,None],linewidth=1.5,legend=True,color=["black","grey","blue"],plot_rainfall=True,plot=None):
    
    #manage date here
    #compute date_range
    date_deb=datetime.datetime.fromisoformat(hdf5["setup"].attrs["start_time"])+duration_to_timedelta(int(hdf5["setup"].attrs["dt"]))
    date_end=datetime.datetime.fromisoformat(hdf5["setup"].attrs["end_time"])+duration_to_timedelta(int(hdf5["setup"].attrs["dt"]))
    date_range=[date_deb,date_end,hdf5["setup"].attrs["dt"]]
    
    # ~ plot=plt.subplots()
    if plot is None:
        if plot_rainfall:
            fig, (ax1, ax2) = plt.subplots(2, 1,height_ratios=[1, 4])
            fig.subplots_adjust(hspace=0)
            plot=[fig, ax2, ax1]
        else:
            fig, ax2 = plt.subplots()
            plot=[fig, ax2]
    else:
        if plot_rainfall:
            fig=plot[0]
            ax1=plot[2]
            ax2=plot[1]
        else:
            fig=plot[0]
            ax2=plot[1]
        
    plot=plot_time_vars(hdf5["input_data/qobs"][:,:],columns=[0],title=title,label="Observations at "+hdf5["mesh/code"][0].decode(),xlabel="Time step",dt=0,ylabel="Discharge $(m^3/s)$",figname=figname,color=color[0],linestyle="--",linewidth=linewidth,xlim=xlim,ylim=ylim,legend=legend,date_range=date_range,plot=plot)
    
    plot=plot_time_vars(hdf5["output/qsim"][:,:],columns=[0],title=title,label="Simulation at "+hdf5["mesh/code"][0].decode(),xlabel="Time step",dt=0,ylabel="Discharge $(m^3/s)$",figname=figname,color=color[1],linestyle="-",linewidth=linewidth,legend=legend,xlim=xlim,ylim=ylim,date_range=date_range,plot=plot)
    
    
    xtics = np.arange(np.datetime64(date_range[0]),np.datetime64(date_range[1]), np.timedelta64(int(date_range[2]), 's'))
    
    if plot_rainfall:
    
        ax1.bar(xtics,hdf5["input_data/mean_prcp"][0,:],label="Average rainfall (mm)")
        ax1.invert_yaxis()
        ax1.grid(alpha=.7, ls="--")
        ax1.get_xaxis().set_visible(False)
        ax1.set_ylim(bottom=1.2*max(hdf5["input_data/mean_prcp"][0,:]))
        ax1.set_ylabel('Average rainfall (mm)')
        
        if legend:
            ax1.legend(loc='upper right')
        else:
            ax1.legend(loc='upper right').set_visible(False)
        
        plot=[fig, ax2, ax1]
    else:
        plot=[fig, ax2]
    
    return plot




def plot_results_assim(res,title="",figname="",columns=[0],xlim=[None,None],linewidth=1.5,linestyle="-",plot=None):
    
    if plot is None:
        plot=[None,None]
    
    
    color = matplotlib.colormaps['gist_rainbow']
    color = matplotlib.colormaps['prism']
    sampling=np.arange(len(res))/len(res)
    i=0
    
    for key,values in res.items():
        
        date_deb=datetime.datetime.fromisoformat(values["setup"].get('start_time'))+duration_to_timedelta([int(values["setup"].get('dt')),'s'])
        date_end=datetime.datetime.fromisoformat(values["setup"].get('end_time'))+duration_to_timedelta(int(values["setup"].get('dt')))
        date_range=[date_deb.strftime("%Y-%m-%d %H:%M"),date_end.strftime("%Y-%m-%d %H:%M"),int(values["setup"].get('dt'))]
        
        plot=plot_time_vars(values["output"].get('qsim'),columns=columns,title="",xlabel="Time step",dt=0,ylabel="Discharge $(m^3/s)$",figname=figname,color=color(sampling[i]),linestyle=linestyle,linewidth=linewidth,legend=True,xlim=[None,None],ylim=[None,None],date_range=date_range,plot=plot)
        
        len_sim=values["output"]['qsim'].shape[1]-1
        y=values["output"]['qsim'][columns,len_sim]
        x=datetime.datetime.fromisoformat(values["setup"].get('end_time'))
        
        fig=plot[0]
        ax=plot[1]
        ax.plot(x,y,marker='o',markersize=12,color=color(sampling[i]))
        
        i=i+1
        
        plot=[fig,ax]
        
        # ~ date_start=datetime.datetime.fromisoformat(values["setup"].get('end_time'))
        # ~ date_range=[date_start.strftime("%Y-%m-%d %H:%M"),date_end.strftime("%Y-%m-%d %H:%M"),int(values["setup"].get('dt'))]
        
        # ~ plot=plot_time_vars(y,title="",label="" + key,xlabel="Time step",dt=0,ylabel="Discharge $(m^3/s)$",figname=figname,marker='o',markersize=8,legend=False,xlim=[None,None],ylim=[None,None],date_range=date_range,plot=plot)
    
    return plot



def plot_results_assim_hdf5(hdf5,title="",figname="",columns=[0],xlim=[None,None],linewidth=1.5,linestyle="-",marker="o",plot=None):
    
    if plot is None:
        plot=[None,None]
    
    
    color = matplotlib.colormaps['gist_rainbow']
    color = matplotlib.colormaps['prism']
    sampling=np.arange(len(list(hdf5.keys())))/len(list(hdf5.keys()))
    i=0
    
    for key in list(hdf5.keys()):
        
        date_deb=datetime.datetime.fromisoformat(hdf5[f"{key}/setup"].attrs['start_time'])+duration_to_timedelta([int(hdf5[f"{key}/setup"].attrs['dt']),'s'])
        date_end=datetime.datetime.fromisoformat(hdf5[f"{key}/setup"].attrs['end_time'])+duration_to_timedelta(int(hdf5[f"{key}/setup"].attrs['dt']))
        date_range=[date_deb.strftime("%Y-%m-%d %H:%M"),date_end.strftime("%Y-%m-%d %H:%M"),int(hdf5[f"{key}/setup"].attrs['dt'])]
        
        plot=plot_time_vars(hdf5[f"{key}/output/qsim"][:,:],columns=columns,title="",xlabel="Time step",dt=0,ylabel="Discharge $(m^3/s)$",figname=figname,color=color(sampling[i]),linestyle=linestyle,linewidth=linewidth,legend=True,xlim=[None,None],ylim=[None,None],date_range=date_range,plot=plot)
        
        len_sim=hdf5[f"{key}/output/qsim"][:,:].shape[1]-1
        y=hdf5[f"{key}/output/qsim"][:,:][columns,len_sim]
        x=datetime.datetime.fromisoformat(hdf5[f"{key}/setup"].attrs['end_time'])
        
        fig=plot[0]
        ax=plot[1]
        ax.plot(x,y,marker=marker,markersize=12,color=color(sampling[i]))
        
        i=i+1
        
        plot=[fig,ax]
    return plot




def plot_results_warmup(result_warmup,title="",figname="",columns=[0],linewidth=1.5,linestyle="-",plot=None):
    
    if plot is None:
        plot=[None,None]
    
    for key,values in result_warmup.items():
        
        date_deb=datetime.datetime.fromisoformat(values["setup"].get('start_time'))+duration_to_timedelta(int(values["setup"].get('dt')))
        date_end=datetime.datetime.fromisoformat(values["setup"].get('end_time'))+duration_to_timedelta(int(values.get('dt')))
        date_range=[date_deb.strftime("%Y-%m-%d %H:%M"),date_end.strftime("%Y-%m-%d %H:%M"),int(values["setup"].get('dt'))]
        
        plot=plot_time_vars(values["output"].get('qsim'),columns=columns,title="",label="W.up for t=" + key,xlabel="Time step",dt=0,ylabel="Discharge $(m^3/s)$",figname=figname,color=color(sampling[i]),linestyle=linestyle,linewidth=linewidth,legend=False,xlim=[None,None],ylim=[None,None],date_range=date_range,plot=plot)
    
    return plot


def plot_results_forecast(result_forecast,title="",figname="",columns=[0],linewidth=1.5,linestyle="-",plot=None):
    
    if plot is None:
        plot=[None,None]
    
    # ~ color = matplotlib.colormaps['gist_rainbow']
    color = matplotlib.colormaps['prism']
    sampling=np.arange(len(result_forecast))/len(result_forecast)
    i=0
    
    for key,values in result_forecast.items():
        
        date_deb=datetime.datetime.fromisoformat(values["setup"].get('start_time'))+duration_to_timedelta(int(values["setup"].get('dt')))
        date_end=datetime.datetime.fromisoformat(values["setup"].get('end_time'))+duration_to_timedelta(int(values["setup"].get('dt')))
        date_range=[date_deb.strftime("%Y-%m-%d %H:%M"),date_end.strftime("%Y-%m-%d %H:%M"),int(values["setup"].get('dt'))]
        
        plot=plot_time_vars(values["output"].get('qsim'),columns=columns,title="",xlabel="Time step",dt=0,ylabel="Discharge $(m^3/s)$",figname=figname,color=color(sampling[i]),linestyle=linestyle,linewidth=linewidth,legend=True,xlim=[None,None],ylim=[None,None],date_range=date_range,plot=plot)
        
        # ~ len_sim=values["output"]['qsim'].shape[1]-1
        y=values["output"]['qsim'][columns,0]
        # ~ x=datetime.datetime.fromisoformat(values["setup"].get('start_time'))
        x=date_deb
        
        fig=plot[0]
        ax=plot[1]
        ax.plot(x,y,marker='X',markersize=12,color=color(sampling[i]))
        
        i=i+1
        
        plot=[fig,ax]
    
    return plot




def plot_time_vars(data,title="",label="",xlabel="",ylabel="",figname="",step=1,columns=[],dx=1.,dt=0.,xlim=[None,None],ylim=[None,None],color="black",linestyle="-",linewidth=1.5,marker='',markersize=4,legend=True,xtics=[],date_range=None,plot=[None,None]):
    
    
    if ((plot[0]!=None) & (plot[1]!=None)):
        fig=plot[0]
        ax=plot[1]
    else:
        fig,ax=plt.subplots()
    
    if (title!=""): ax.set_title(title)
    if (xlabel!=""): ax.axes.set_xlabel(xlabel)
    if (ylabel!=""): ax.axes.set_ylabel(ylabel)
    
    if (len(xtics)==0):
        xtics=np.arange(0,data.shape[1])
        if (dt>0):
            xtics=xtics*dt
    
    if date_range is not None:
        xtics = np.arange(np.datetime64(date_range[0]),np.datetime64(date_range[1]), np.timedelta64(int(date_range[2]), 's'))
    
    if (len(columns)>0):
        for i in columns:
            ax.plot(xtics[:],data[i,:],color=color,label=label,ls=linestyle,lw=linewidth,marker=marker,markersize=markersize)
    else:
        for i in range(0,data.shape[0],step):
            ax.plot(xtics[:],data[i,:],label=label,ls=linestyle,lw=linewidth,marker=marker,markersize=markersize)
    
    if (ylim[0]!=None):
        ax.set_ylim(bottom=ylim[0])
    if (ylim[1]!=None):
        ax.set_ylim(top=ylim[1])
    if (xlim[0]!=None):
        ax.set_xlim(left=xlim[0])
    if (xlim[1]!=None):
        ax.set_xlim(right=xlim[1])
    
    ax.axes.grid(True,alpha=.7, ls="--")
    if (legend):
        ax.legend(loc='upper left')
    else:
        ax.legend(loc='upper left').set_visible(False)
    
    if (len(figname)>0):
        fig.savefig(figname, transparent=False, dpi=80, bbox_inches="tight")
    # ~ else:
        # ~ fig.show()
    
    plot=[fig,ax]
    
    return plot



def save_figure(fig,figname="myfigure",xsize=8,ysize=6,transparent=False,dpi=80):
    fig.set_size_inches(xsize, ysize, forward=True)
    fig.savefig(figname, transparent=transparent, dpi=dpi, bbox_inches="tight")


def save_figure_from_plot(plot,figname="myfigure",xsize=8,ysize=6,transparent=False,dpi=80,xlim=[None,None],ylim=[None,None]):
    
    fig=plot[0]
    ax=plot[1]
    
    if (ylim[0]!=None):
        ax.set_ylim(bottom=ylim[0])
    if (ylim[1]!=None):
        ax.set_ylim(top=ylim[1])
    if (xlim[0]!=None):
        ax.set_xlim(left=xlim[0])
    if (xlim[1]!=None):
        ax.set_xlim(right=xlim[1])
    
    fig.set_size_inches(xsize, ysize, forward=True)
    fig.savefig(figname, transparent=transparent, dpi=dpi, bbox_inches="tight")



def plot_matrix(matrix,mask=None,figname="",title="",label="",vmin=None,vmax=None):
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    
    if mask is not None:
        ma = (mask == 0)
        ma_var = np.where(ma, np.nan, matrix)
    else:
        ma_var=matrix
    
    map_var = ax.imshow(ma_var,vmin=vmin,vmax=vmax);
    fig.colorbar(map_var, ax=ax, label=label,shrink=0.75);
    
    plot=[fig,ax]
    
    if (len(figname)>0):
        fig.savefig(figname, transparent=False, dpi=80, bbox_inches="tight")
    else:
        fig.show()
    
    plot=[fig,ax]
    
    return plot



def plot_image(matrice=np.zeros(shape=(2,2)),bbox=None,title="",xlabel="",ylabel="",zlabel="",vmin=None,vmax=None,mask=None,figname=""):
    """
    Function for plotting a matrix as an image
    
    Parameters
    ----------
    matrice : numpy array
    bbox : ["left","right","bottom","top"] bouding box to put x and y coordinates instead of the shape of the matrix
    title : character, title of the plot
    xlabel : character, label of the xaxis
    ylabel : character, label of the y axis
    zlabel : character, label of the z axis
    vmin: real, minimum z value
    vmax: real, maximum z value
    mask: integer, matrix, shape of matice, contain 0 for pixels that should not be plotted
    show: booloen, true call fig.show() or false return fig instead.
    
    Examples
    ----------
    smash.utils.plot_image(mesh_france['drained_area'],bbox=bbox,title="Surfaces drainées",xlabel="Longitude",ylabel="Latitude",zlabel="Surfaces drainées km^2",vmin=0.0,vmax=1000,mask=mesh_france['global_active_cell'])

    """
    
    matrice=np.float32(matrice)
    
    if (type(bbox)!=type(None)):
        extend=[bbox["left"],bbox["right"],bbox["bottom"],bbox["top"]]#bbox.values()
    else:
        extend=None
    
    if (type(mask)!=type(None)):
        matrice[np.where(mask==0)]=np.nan
    
    # ~ color_matrice=matrice
    # ~ if vmax!=None:
        # ~ color_matrice[np.where(matrice>vmax)]=vmax
    # ~ if vmin!=None:
        # ~ color_matrice[np.where(matrice<vmin)]=vmin
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    im=ax.imshow(matrice,extent=extend,vmin=vmin,vmax=vmax)
    ax.axes.set_xlabel(xlabel)
    ax.axes.set_ylabel(ylabel)
    
    plt.colorbar(im)
    #fig.colorbar(plt.cm.ScalarMappable(norm=None), ax=ax, label=zlabel)
    
    if (len(figname)>0):
        fig.savefig(figname, transparent=False, dpi=80, bbox_inches="tight")
    else:
        fig.show()
    





def plot_model_params_and_states(model,variables,fstates=False):
    
    if not isinstance(variables, list):
        raise ValueError(
            f"variables '{variables}' must be list of parameters or states names"
            )
    
    nb_subplot=len(variables)
    if (nb_subplot>1):
        nb_rows=math.ceil(math.sqrt(nb_subplot))
        nb_cols=math.ceil(nb_subplot/nb_rows)
        #nb_cols=nb_subplot- math.floor(math.sqrt(nb_subplot))
    else:
        nb_rows=1
        nb_cols=1
    
    print(nb_rows,nb_cols)
    fig, ax = plt.subplots(nb_rows, nb_cols)
    
    if len(variables)==1:
        ax = [ax]
    
    fig.suptitle(f'Optimized parameter set')
    
    for i,var in enumerate(variables):
        
        rr=(i+1)/(nb_cols)
        part_entiere=math.floor(rr)
        part_reel=rr-part_entiere
        
        if part_reel>0:
            r=max(0,part_entiere)
        else:
            r=max(0,part_entiere-1)
            
        if (part_reel==0.):
            c=nb_cols-1
        else:
            c=math.ceil((part_reel)*(nb_cols))-1
        
        #r=math.ceil(i/(nb_cols))
        #c=(r*nb_cols-i)
        print(i,r,c)
        
        if isinstance(model,dict):
            
            for key,list_param in smash.core._constant.STRUCTURE_PARAMETERS.items():
                
                if var in list_param:
                    
                    values=model["parameters"][var]
                    break
            
            for key,list_states in smash.core._constant.STRUCTURE_STATES.items():
                
                if var in list_states:
                    
                    if fstates==True :
                        values=model["output"]["fstates"][var]
                    else:
                        values=model["states"][var]
                    
                    break
            ma = (model["mesh"]["active_cell"] == 0)
            
        else:
            
            if var in model.setup._parameters_name:
                
                values=getattr(model.parameters,var)
            
            if var in model.setup._states_name:
                
                if fstates:
                    values=getattr(model.output.states,var)
                else:
                    values=getattr(model.states,var)
            
            ma = (model.mesh.active_cell == 0)
        
        ma_var = np.where(ma, np.nan, values)
        
        map_var = ax[r,c].imshow(ma_var);
        fig.colorbar(map_var, ax=ax[r,c], label=var,shrink=0.75);
    
    plot=[fig,ax]
    return plot



def plot_lcurve(instance,figname=None,transform=False,annotate=True,plot=None):
    
    if not isinstance(instance,dict):
        raise ValueError(
            f"instance must be a dict"
            )
    
    if plot is not None:
        fig=plot[0]
        ax=plot[1]
    else:
        fig,ax=plt.subplots()
    
    if "wjreg_lcurve_opt" in instance:
        pass
    else:
        return plot
    
    if (transform==True):
        
        jobs_max=np.zeros(shape=len(instance["cost_jobs"]))
        jobs_max[:]=instance["cost_jobs_initial"]
        
        jobs_max[:]=max(instance["cost_jobs"])
        
        jobs_min=min(instance["cost_jobs"])
        jreg_max=max(instance["cost_jreg"])
        
        #index_min=np.where(instance["cost_jobs"] == jobs_min)
        
        #choose the lower value of jreg if index_min has many values
        #index_jreg_max=list(instance["cost_jreg"]).index(min(instance["cost_jreg"][index_min[0]]))
        #jreg_max=instance["cost_jreg"][index_jreg_max]
        
        
        jreg_min=np.zeros(shape=len(instance["cost_jreg"]))
        #jreg_min[:]=instance["cost_jreg_initial"]
        #si cost_jreg_initial > 0 then prendre :
        jreg_min[:]=min(instance["cost_jreg"])
        
        go_plot=False
        if (np.all((jobs_max[0]-jobs_min)>0.)) and (np.all((jreg_max-jreg_min[0])>0.)):
            xs=(jobs_max-instance["cost_jobs"])/(jobs_max[0]-jobs_min)
            ys=(instance["cost_jreg"]-jreg_min)/(jreg_max-jreg_min[0])
            go_plot=True
            
        
        # ~ if (np.all((jreg_max-jreg_min[0])>0.)):
            # ~ ys=(instance["cost_jreg"]-jreg_min)/(jreg_max-jreg_min[0])
        
        # ~ #plot lcurve 
        if (go_plot):
            ax.plot(xs,ys, ls="--", marker="x", color="grey");
            
            # zip joins x and y coordinates in pairs
            i=0
            for x,y in zip(xs,ys):
                
                label=""
                textcolor="black"
                point_type="."
                ax.plot(x,y, color=textcolor,marker=point_type,markersize=5);
                
                if (instance["wjreg"][i]==instance["wjreg_lcurve_opt"]):
                    textcolor="red"
                    point_type="o"
                    ax.plot(x,y, color=textcolor,marker=point_type,markersize=8);
                    
                    label = "{:.2E}".format(instance["wjreg_lcurve_opt"])
                
                #print(instance["wjreg"][i],instance["wjreg_fast"])
                
                go_plot=False
                if (instance["wjreg"][i]==instance["wjreg_fast"]):
                    go_plot=True
                elif (abs(1.-instance["wjreg"][i]/instance["wjreg_fast"])<0.0001):
                    go_plot=True
                
                if (go_plot) :
                    textcolor="green"
                    point_type="^"
                    ax.plot(x,y, color=textcolor,marker=point_type,markersize=8);
                
                if annotate:
                    ax.annotate(label, # this is the text
                                 (x,y), # these are the coordinates to position the label
                                 textcoords="offset points", # how to position the text
                                 xytext=(0,5), # distance from text to points (x,y)
                                 ha='right', # horizontal alignment can be left, right or center
                                 color=textcolor, fontsize=10) 
                
                i=i+1
            
        ax.plot([0,1],[0,1],color="red")
        
    else:
        
        ax.plot(instance["cost_jobs"],instance["cost_jreg"], ls="--", marker="x",color="grey");
        
        # zip joins x and y coordinates in pairs
        i=0
        for x,y in zip(instance["cost_jobs"],instance["cost_jreg"]):
            
            label = "{:.2E}".format(instance["wjreg"][i])
            textcolor="black"
            
            if (instance["wjreg"][i]==instance["wjreg_lcurve_opt"]):
                textcolor="red"
            
            if (abs(1.-instance["wjreg"][i]/instance["wjreg_fast"])<0.0001):
                textcolor="green"
            
            if annotate:
                ax.annotate(label, # this is the text
                             (x,y), # these are the coordinates to position the label
                             textcoords="offset points", # how to position the text
                             xytext=(0,5), # distance from text to points (x,y)
                             ha='right', # horizontal alignment can be left, right or center
                             color=textcolor, fontsize=10) 
            
            i=i+1
        
    
    ax.set_xlabel("(jobs_max-jobs)/(jobs_max_jobs_min)");
    ax.set_ylabel("(jreg-jreg_min)/(jreg_max-jreg_min)");
    
    if figname is not None:
        fig.savefig(figname, transparent=False, dpi=80, bbox_inches="tight")
    
    plot=[fig,ax]
    return plot


def plot_dist_wjreg(res_assim):
    
    fig,ax=plt.subplots()
    x=list()
    y=list()
    for key,values in res_assim.items():
        lcurve=values["lcurve"]
        color="black"
        point_type="."
        markersize=6
        if (lcurve["wjreg_lcurve_opt"] is not None) and (lcurve["wjreg_fast"] is not None) and (lcurve["wjreg_lcurve_opt"] >0.) and (lcurve["wjreg_fast"] >0.):
            x.append(float(lcurve["wjreg_lcurve_opt"]))
            y.append(float(lcurve["wjreg_fast"]))
    
    xn=np.log(np.array(x))
    yn=np.log(np.array(y))
    
    ax.scatter(np.array(xn),np.array(yn), color=color,marker=point_type)
    ax.set_xlabel("log(wjreg) - Lcurve method");
    ax.set_ylabel("log(wjreg) - Fast method");
    
    min_val=min(min(xn),min(yn))
    max_val=max(max(xn),max(yn))
    x=np.arange(min_val,max_val+1)
    y=np.arange(min_val,max_val+1)
    ax.plot(x,y, color="red",marker=None,markersize=markersize)
    
    return fig,ax




def plot_mesh(model=None,mesh=None,title=None,figname=None,coef_hydro=99.):
    
    if model is not None:
        if isinstance(mesh_in,smash.Model):
            mesh=model.mesh
        else:
            raise ValueError(
                f"model object must be an instance of smash Model"
                )
    elif mesh is not None:
        if isinstance(mesh,dict):
            pass
        else:
            raise ValueError(
                f"mesh must be a dict"
                )
    else:
        raise ValueError(
                f"model or mesh are mandatory and must be a dict or a smash Model object"
                )
    
    mesh["active_cell"]
    gauge=mesh["gauge_pos"]
    stations=mesh["code"]
    flow_acc=mesh["flwacc"]
    
    na = (mesh["active_cell"] == 0)
    
    flow_accum_bv = np.where(na, 0., flow_acc.data)
    surfmin=(1.-coef_hydro/100.)*np.max(flow_accum_bv)
    mask_flow=(flow_accum_bv < surfmin)
    flow_plot=np.where(mask_flow, np.nan,flow_accum_bv.data)
    flow_plot=np.where(na, np.nan,flow_plot)
    
    fig, ax = plt.subplots()
    
    if title is not None:
        ax.set_title(title)
    
    active_cell = np.where(na, np.nan, mesh["active_cell"])
    #cmap = ListedColormap(["grey", "lightgray"])
    cmap = ListedColormap([ "lightgray"])
    ax.imshow(active_cell,cmap=cmap)
    
    #cmap = ListedColormap(["lightblue","blue","darkblue"])
    myblues = matplotlib.colormaps['Blues']
    cmp = ListedColormap(myblues(np.linspace(0.30, 1.0, 265)))
    im=ax.imshow(flow_plot,cmap=cmp)
    #im=ax.imshow(flow_plot,cmap="Blues")
    
    fig.colorbar(im,cmap="Blues", ax=ax, label="Cumulated surface (km²)",shrink=0.75);
    
    
    for i in range(len(stations)):
        coord=gauge[i]
        code=stations[i]
        ax.plot(coord[1],coord[0], color="green",marker='o',markersize=6)
        ax.annotate(code, # this is the text
                     (coord[1],coord[0]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,5), # distance from text to points (x,y)
                     ha='right', # horizontal alignment can be left, right or center
                     color="red", 
                     fontsize=10) 
    
    if figname is not None:
        fig.savefig(figname, transparent=False, dpi=80, bbox_inches="tight")
    
    return fig,ax




def plot_event_seg(model,event_seg,code=''):
    event_seg_sta_aval = event_seg[(event_seg['code'] == code)]
    
    dti = pd.date_range(start=model.setup.start_time, end=model.setup.end_time, freq="H")[1:]
    qo = model.input_data.qobs[0, :]
    prcp = model.input_data.mean_prcp[0, :]
    starts = pd.to_datetime(event_seg_sta_aval["start"])
    ends = pd.to_datetime(event_seg_sta_aval["end"])
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0)
    ax1.bar(dti, prcp, color="lightslategrey", label="Rainfall");
    
    ax1.axvspan(starts[0], ends[0], alpha=.1, color="red", label="Event segmentation");
    for i in range(1,len(starts)):
        ax1.axvspan(starts[i], ends[i], alpha=.1, color="red");
        ax1.axvspan(starts[i], ends[i], alpha=.1, color="red");
    
    ax1.grid(alpha=.7, ls="--")
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel("$mm$");
    ax1.invert_yaxis()
    ax2.plot(dti, qo, label="Observed discharge");
    for i in range(0,len(starts)):
        ax2.axvspan(starts[i], ends[i], alpha=.1, color="red");
    
    ax2.grid(alpha=.7, ls="--")
    ax2.tick_params(axis="x", labelrotation=20)
    ax2.set_ylabel("$m^3/s$");
    ax2.set_xlim(ax1.get_xlim());
    fig.legend();
    fig.suptitle("V5014010");
    
    return fig



import xarray as xr
import numpy as np
import pandas as pd
import h5py
from analysis_tools import *

# import functions
#------------------------------------------

def strip_H5_to_dataset(every_analysis_output_dict, end_list, last_lvl, lvl_h5_dataset):
    '''
    Get recursively the structure and data of an exported .h5 SIMA simulation file, and put it nicely in a xarray structure with dimensions [model, condition, analysis, time].
    If the recquired output are not present for some of the model, conditions or analysis of the SIMA run, it is avoided. 

    INPUTS : 
        * every_analysis_output_dict    : dictionary {sima_path:output_renaming} linking the sima path (necessary to access an output data from a SIMA run .h5 output) to the new name we want to give to this output, more easily understandable (and shorter to use!)
        * end_list                      : a list that contains the data we are collecting
        * last_lvl                      : three element list containing the keys to the last three level of the h5 dataset we have dived in
        * lvl_h5_dataset                : the h5 dataset at the current level
    OUTPUT
        * ds (or None)                  : a xarray.Dataset with dimensions [model, condition, analysis, time] containing different xarray.Datarrays representing the different output we asked
    '''
    keys = lvl_h5_dataset.keys() # The keys of the level we are in
    # If we are still not to the depth-level where there is condition run data 
    if not (({'Dynamic'} & set(keys)) or ({'Static'} & set(keys)) or ({'Eigenvalue'} & set(keys))):
        last_lvl = last_lvl[1:] + [None] # Preparing our level list to host new entries by removing the oldest level
        res = []
        for key in keys:
            if key != "Variables":
                last_lvl[-1] = key # Updating with a fresh level
                res.append(strip_H5_to_dataset(every_analysis_output_dict, end_list, last_lvl, lvl_h5_dataset[key])) # Recursively running the function to delve deeper in the h5 dataset and getting the results in "res" list
        if len(res) > 0: 
            end_list.extend(res) # In case we got some datas, list isn't empty and the "end_list" gets extended
        return None # The functions returns itself no data
    else:
        # Situation where we have datas from different analysis
        # Then, the keys must contain "Variables" and also other names such as "Dynamic", "Static" for the different analysis that has been run
        var_n_an_keys = list(lvl_h5_dataset.keys()) # Variables and Analysis keys
        metadata = lvl_h5_dataset["Variables"] # The variables of the SIMA condition run are stored as metadata in the data structure we will be using
        # Putting the metadata in a dictionary format
        attrs = {}
        for key,item in metadata.items():
            attrs[key] = item[()]
        metadata = attrs
        var_n_an_keys.remove('Variables')
        dict_vars = {} # Dictionnary collecting data in the form of xarray.DataArray
        dict_coords = {} # Coordinates for the data we collect, answering to the dimension recquirement of the data structure ['model', 'condition', 'analysis', 'time']
        dict_coords['model'] = [last_lvl[0]] # model is always the 3rd last level we've been through in case of a condition Set/Space SIMA simulation
        dict_coords['condition'] = [last_lvl[-1]] # condition run name is always the upper level of the "data level"
        dict_coords['analysis'] = [] # We will get through the different analysis run
        for analysis in var_n_an_keys:
            dict_coords['analysis'] += [analysis] 
            for path, output in every_analysis_output_dict.items(): # Getting through the recquired variables we need
                # If the variable is present for this analysis, we get it. Otherwise, we get nan values for it and a nice error message tells us it hasn't been found
                try:
                    test = lvl_h5_dataset[analysis][path] 
                except :
                    print("No ", output, " found for condition ", last_lvl[-1], 'on model ', last_lvl[0], 'with analysis ', analysis, '.')
                else : 
                    series = lvl_h5_dataset[analysis][path]
                    # Constructing time from serie
                    to = series.attrs['start'][()]
                    dt = series.attrs['delta'][()]
                    N = len(series)
                    te = to + (N-1)*dt
                    time = np.arange(to,te+dt,dt)
                    # Constructing our data structure for one of the outputs
                    ds = xr.DataArray(
                        name = output,
                        data = [[[pd.Series(series)]]], # datas are transformed to pandas series
                        coords = {
                            'model':('model', [last_lvl[0]]),
                            'condition':('condition',[last_lvl[-1]]),
                            'analysis':('analysis', [analysis]),
                            'time':('time', time)
                        },
                        dims = ['model', 'condition', 'analysis', 'time'],
                        attrs = metadata
                    )
                    dict_vars[output] = ds # Collecting newly created dataarray to our dataset dictionary
                    dict_coords['time'] = time # Collecting the time, in case the different outputs are not stored with the same time basis 

        # Constructing a xarray.Dataset from the different output xarray.DataArray we collected
        dset = xr.Dataset(
            data_vars = dict_vars,
            coords = dict_coords,
            attrs=metadata
        )
        # Returning the dataset
        return(dset)

def dataset_from_h5(h5_file, keys_dict):
    '''
    Initialize the strip_H5_to_dataset() function.
    Takes a path to h5 file and a dictionary {sima_path:output_renaming} and returns an xarray.Dataset with dimensions [model, condition, analysis, time].
    '''
    end_list = []
    h5_dataset = h5py.File(h5_file)
    lvls = [list(h5_dataset.keys())[0]]*3

    strip_H5_to_dataset(keys_dict,end_list, lvls, h5_dataset)

    final = end_list[0]

    for ds in end_list[1:]:
        final = final.combine_first(ds) # Merging of all the dataset collected with strip_H5_to_dataset() function

    print('==============================\n SUCCESS IMPORTING DATASET ')
    print('------------------------------\n THIS IS THE IMPORTED DATASET')
    print(final)
    print('-------------------------------\n SUCCESS IMPORTING AND MERGING DATASET \n-------------------------------')
    return final

##########################################
# Begin CLASS
##########################################


class SIMHigh5():
    # Initializing the class can be done by a filename, a xarray dataset or a class instance
    def __init__(self, h5_data, keys_dict, source = "file", name="NewDataset"):
        '''
        SIMHigh5 is an xarray wrapper for SIMA run data loaded under h5 format. 
        Main feature is every-SIMA_workflow adaptative load function that can delve recursively into every h5 SIMA datafile and returns it into easier and understandable format.
        It comes with numerous methods to process the data.
        '''
        self.dims = ['model', 'condition', 'analysis', 'time']
        if source == "file":
            self.df = dataset_from_h5(h5_data, keys_dict)
        elif source == "SIMHigh5":
            self.df = h5_data.df
        elif source == "xarray":
            self.df = h5_data
        self.name = name

    def selection(self, sel_dict):
        '''
        Return a new class instance whose dataset is a selection of the former regarding the conditions contained in the sel_dict dictionary 
        '''
        new_ds = self.df.sel(sel_dict)
        return SIMHigh5(new_ds, None, source = "xarray")
    
    def extract_run(self, dict_coords, show = True):
        '''
        Returns a panda DataFrame and the metadata dict of a SIMA run, according to the dict_coords {model: mmmmm , condition:ccccc, analysis: aaaaa} triplet.
        '''
        run = self.df.sel(dict_coords)
        var = run.attrs
        run = run.to_pandas()
        if show:
            print('\n----------Successfully extracted run to panda.Dataframe structure-----------\n')
            print(run.describe())
        return run, var
    
    def extract_dataframe(self, show = True):
        pd_dataframe = self.df.to_dataframe()
        if show:
            print('\n----------Successfully extracted all datas to panda.Dataframe structure-----------\n')
            print(pd_dataframe.head())
            print(pd_dataframe.describe())
        return pd_dataframe

    def merge(self, list_of_H5DF):
        '''
        Merges the dataset in the list to the original dataset. Returns None.
        '''
        for H in list_of_H5DF:
            self.df = self.df.combine_first(H.df)
        return None

    def skip_transient(self, T_trans):
        '''
        Returns a new class instance whose timeseries datas where chopped to match t > T_trans.
        '''
        time = self.df.coords['time'].values
        t_idx = time>T_trans
        time = time[t_idx]
        new_df = self.df.sel({'time': time})
        print(new_df)
        return SIMHigh5(new_df, None, source="xarray")

    def timeserie(self, dict_coords, output, show=True):
        '''
        Returns tuple (time, serie) from a given output and coordinates.
        '''
        array = self.df.sel(dict_coords)[output]
        time = (array.coords['time'].values)
        serie = pd.Series(data = array.values, name = output, index = time)
        if show:
            print(serie.describe())
        return time, serie
    
    def comparePlot_timeseries(self, ax, dim_in_legend,  dict_other_fixed_coord, output, legend_list=[], color_list=[]):
        '''
        Given an ax, an output, a variable dimension and specifying the fixed dimensions with selecting their coordinates, plots the timeseries of the recquired output for all the coordinates in the varying dimension.
        INPUTS : 
            * ax                        : a matplotlib.pyplot ax
            * dim_in_legend             : either 'model', 'condition' or 'analysis'
            * dict_other_fixed_coord    : dictionary that specify the two other dimension coordinates ; for instance, if dim_in_legend = 'condition', it is {'model':name_of_model, 'analysis':name_of_analysis}
            * output                    : string standing for the output
        OUTPUT
            * NONE
        '''
        if len(legend_list) == 0:
            l = []
        for i,coord in enumerate(self.df.coords[dim_in_legend].values):
            dict_coords = {dim_in_legend:coord}
            dict_coords.update(dict_other_fixed_coord)
            time, serie = self.timeserie(dict_coords, output, show = False)
            if len(color_list) > 0:
                ax.plot(time, serie, color = color_list[i])
            else:
                ax.plot(time, serie)
            if len(legend_list) == 0:
                l.append(coord)
        if len(legend_list) > 0:
            ax.legend(legend_list)
        else:
            ax.legend(l)
        ax.set_ylabel(output)
        ax.set_xlabel('Time (s)')
        return None
    
##########################################
# End
##########################################


# Tests
#------------------------------------------


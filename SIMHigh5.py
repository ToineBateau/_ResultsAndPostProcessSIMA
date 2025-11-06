import xarray as xr
import numpy as np
import pandas as pd
import h5py
from analysis_tools import *


# import functions
#------------------------------------------
def ensure_list(ds,coord):
    '''
    Ensures that when coordinates are extracted, it results in a list.
    '''
    val = ds.coords[coord].values
    if val.size > 1:
        return list(val)
    else:
        return [str(val)]
    
def drop_constant_vars(ds: xr.Dataset) -> xr.Dataset:
    '''
    Drop variables that have constant values across all dimensions.
    '''
    varying_vars = []
    for var in ds.data_vars:
        values = ds[var].values
        try:
            # Handle different types appropriately
            if values.dtype == np.dtype('O') or values.dtype.kind == 'S' or values.dtype.kind == 'U':
                # For object, string or unicode arrays
                n_unique = len(set(values.flatten()))
            else:
                # For numeric arrays
                n_unique = np.unique(values).size
            
            if n_unique > 1:
                varying_vars.append(var)
        except TypeError:
            # If we can't determine uniqueness, keep the variable
            varying_vars.append(var)
            
    return ds[varying_vars]

def strip_H5_to_dataset(every_analysis_output_dict, end_list, last_lvl, lvl_h5_dataset, debug):
    '''
    Get recursively the structure and data of an exported .h5 SIMA simulation file, and put it nicely in a xarray structure with dimensions [model, condition, analysis, time].
    If the recquired output are not present for some of the model, conditions or analysis of the SIMA run, it is avoided. 

    INPUTS : 
        * every_analysis_output_dict    : dictionary {sima_path:output_renaming} linking the sima path (necessary to access an output data from a SIMA run .h5 output) to the new name we want to give to this output, more easily understandable (and shorter to use!)
        * end_list                      : a list that contains the data we are collecting
        * last_lvl                      : three element list containing the keys to the last three level of the h5 dataset we have dived in
        * lvl_h5_dataset                : the h5 dataset at the current level
        * debug                         : flag allowing printing of relevant logs
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
                res.append(strip_H5_to_dataset(every_analysis_output_dict, end_list, last_lvl, lvl_h5_dataset[key], debug)) # Recursively running the function to delve deeper in the h5 dataset and getting the results in "res" list
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
            var = item[()]
            attrs[key] = var
        metadata = attrs
        var_n_an_keys.remove('Variables')
        dict_vars = {} # Dictionnary collecting data in the form of xarray.DataArray
        dict_coords = {} # Coordinates for the data we collect, answering to the dimension recquirement of the data structure ['model', 'condition', 'analysis', 'time']
        dict_coords['model'] = [last_lvl[0]] # model is always the 3rd last level we've been through in case of a condition Set/Space SIMA simulation
        dict_coords['condition'] = [last_lvl[-1]] # condition run name is always the upper level of the "data level"
        dict_coords['analysis'] = [] # We will get through the different analysis run
        dict_coords['time'] = [] # The different timespans, that can be different among outputs, are stored for the final dataset
        for analysis in var_n_an_keys:
            dict_coords['analysis'] += [analysis] 
            for path, output in every_analysis_output_dict.items(): # Getting through the recquired variables we need
                # If the variable is present for this analysis, we get it. Otherwise, we get nan values for it and a nice error message tells us it hasn't been found
                try:
                    test = lvl_h5_dataset[analysis][path] 
                except :
                    if debug:
                        print("No ", output, " found for condition ", last_lvl[-1], 'on model ', last_lvl[0], 'with analysis ', analysis, '.') # Message telling whether the recquired output is present in the dataset for the specified [model, condition, analysis]
                else : 
                    series = lvl_h5_dataset[analysis][path]
                    # Constructing time from serie
                    to = float(series.attrs['start'][()])
                    dt = float(series.attrs['delta'][()])
                    N = len(series)
                    te = to + (N-1)*dt
                    time = np.arange(to, te+dt, dt)
                    # Store raw numeric data (not pandas Series) so we can interpolate later
                    vals = np.array(series[:], dtype=float)
                    # Constructing our data structure for one of the outputs
                    ds = xr.DataArray(
                        name=output,
                        data=np.array([[[vals]]]),  # shape (model=1, condition=1, analysis=1, time=N)
                        coords={
                            'model': ('model', [last_lvl[0]]),
                            'condition': ('condition', [last_lvl[-1]]),
                            'analysis': ('analysis', [analysis]),
                            'time': ('time', time)
                        },
                        dims=['model', 'condition', 'analysis',  'time'],
                        attrs=metadata
                    )
                    dict_vars[output] = ds  # Collecting newly created dataarray to our dataset dictionary
                    dict_coords['time'] += [time]  # Collecting the time, in case the different outputs are not stored with the same time basis

        # Build a common, finest time grid and linearly interpolate other series to this grid
        # Determine finest dt among collected times
        all_times = dict_coords['time']
        if len(all_times) > 0:
            min_dt = np.min([np.min(np.diff(t)) for t in all_times if len(t) > 1])
            start = np.min([t[0] for t in all_times])
            end = np.max([t[-1] for t in all_times])
            # Create common time using finest dt
            common_time = np.arange(start, end, min_dt)
        else:
            common_time = np.array([])

        # Interpolate each DataArray onto common_time (linear). For values outside original range, use nearest fill (ffill/bfill).
        for key, da in dict_vars.items():
            orig_time = da.coords['time'].values
            orig_vals = da.values.reshape(-1, orig_time.size)[0]
            if common_time.size == 0:
                # nothing to do
                interp_vals = orig_vals
                new_time = orig_time
            else:
                # Use numpy.interp for reliable float-index interpolation and better performance.
                # np.interp performs linear interpolation and accepts float arrays.
                # For values outside the original range, set to nearest original value (left/right).
                # Ensure orig_time and orig_vals are 1D numpy arrays
                t0 = np.asarray(orig_time, dtype=float)
                v0 = np.asarray(orig_vals, dtype=float)
                # np.interp requires x to be increasing; orig_time should already be increasing.
                interp_vals = np.interp(common_time, t0, v0, left=v0[0], right=v0[-1])
                new_time = common_time

            # Replace DataArray data and time coord
            new_da = xr.DataArray(
                name=da.name,
                data=np.array([[[interp_vals]]]),
                coords={
                    'model': ('model', [last_lvl[0]]),
                    'condition': ('condition', [last_lvl[-1]]),
                    'analysis': ('analysis', [da.coords['analysis'].values[0]]),
                    'time': ('time', new_time)
                },
                dims=['model', 'condition', 'analysis', 'time'],
                attrs=da.attrs
            )
            dict_vars[key] = new_da

        # Constructing a xarray.Dataset from the different output xarray.DataArray we collected
        dset = xr.Dataset(
            data_vars=dict_vars,
            coords={'model': dict_coords['model'], 'condition': dict_coords['condition'], 'analysis': dict_coords['analysis'], 'time': common_time},
            attrs = metadata
        )
        # Returning the dataset
        return(dset)

def dataset_from_h5(h5_file, keys_dict, debug):
    '''
    Initialize the strip_H5_to_dataset() function.
    Takes a path to h5 file and a dictionary {sima_path:output_renaming} and returns an xarray.Dataset with dimensions [model, condition, analysis, time].
    '''
    # Retrieving data from H5 file and putting it into DATA list of xr.Dataset
    end_list = []
    h5_dataset = h5py.File(h5_file)
    lvls = [list(h5_dataset.keys())[0]]*3

    strip_H5_to_dataset(keys_dict,end_list, lvls, h5_dataset, debug)

    end_list = [ds for ds in end_list if ds is not None] # Suppression of none values

    # Building METADATA datasets, with dims [model, condition] same than the DATA datasets, and putting them into list
    attrs_list = []
    for ds in end_list:
        attrs_dict = {}
        attrs_coords = {
        'model':[],
        'condition':[]
        }
        coords_da = {}
        for key, coord in ds.coords.items():
            if (key not in ['time', 'analysis']):
                attrs_coords[key] = attrs_coords[key] + list(coord.values)
                coords_da[key] = list(coord.values)
        for key, val in ds.attrs.items(): 
            attrs_dict[key] = xr.DataArray(
                    data = [[val]],
                    coords=coords_da,
                    dims=['model', 'condition']
            )
        attrs_list.append(
            xr.Dataset(
                data_vars=attrs_dict,
                coords=attrs_coords
            )
        )
    
    # Merging the datasets into one DATA dataset and one METADATA dataset
    metadata = xr.merge(attrs_list, join= 'outer', compat='no_conflicts')
    final = xr.merge(end_list, join= 'outer', compat='no_conflicts', combine_attrs='drop') # Merging of all the dataset collected with strip_H5_to_dataset() function

    return final,metadata

def xr_selection(df, sel_dict):
    """
    Safely select data from an xarray Dataset using sel_dict.
    Only applies selection if keys exist in df.indexes.
    
    Parameters:
        df (xr.Dataset): The dataset.
        sel_dict (dict): Dictionary of coordinates to select.
    
    Returns:
        (bool, xr.Dataset or None): (True, selected dataset) if selection applied,
                                    (False, None) otherwise.
    """
    # Get valid keys from indexes (iterable coordinates)
    valid_keys = set(df.indexes.keys())
    
    # Filter selection dictionary
    filtered_sel = {k: v for k, v in sel_dict.items() if k in valid_keys}
    
    if filtered_sel:
        return True, df.sel(filtered_sel)
    else:
        return False, None

##########################################
# Begin CLASS
##########################################


class SIMHigh5():
    # Initializing the class can be done by a filename, a xarray dataset or a class instance
    def __init__(self, h5_data, keys_dict, source = "file", name="NewDataset", metadata=None, silent = True, debug = True):
        '''
        SIMHigh5 is an xarray wrapper for SIMA run data loaded under h5 format. 
        Main feature is every-SIMA_workflow adaptative load function that can delve recursively into every h5 SIMA datafile and returns it into easier and understandable format.
        It comes with numerous methods to process the data.
        '''
        self.dims = ['model', 'condition', 'analysis', 'variables', 'time']
        if source == "file":
            self.df, self.metadata = dataset_from_h5(h5_data, keys_dict, debug)
        elif source == "SIMHigh5":
            self.df = h5_data.df
            self.metadata = h5_data.metadata
        elif source == "xarray":
            self.df = h5_data
            self.metadata = metadata
        self.name = name

        self.models = ensure_list(self.df, 'model')
        self.conds = ensure_list(self.df, 'condition')

        self.varying_metadata = drop_constant_vars(self.metadata)

        if not silent:
            print('==============================\n SUCCESS IMPORTING DATASET ')
            print('------------------------------\n THIS IS THE IMPORTED DATASET')
            print(self.df)
            print('***************************\n THIS IS THE METADATA DATASET')
            print(self.metadata)
            print('+++++++++++++++++++++++++++\n THIS IS THE VARYING METADATA')
            print(self.varying_metadata)
            print('-------------------------------\n ****** SUCCESS ******\n==============================')

    def selection(self, sel_dict):
        '''
        Return a new class instance whose dataset is a selection of the former regarding the conditions contained in the sel_dict dictionary 
        '''
        flag, new_ds = xr_selection(self.df, sel_dict)
        if flag :    
            return SIMHigh5(new_ds, None, source = "xarray", name = self.name, metadata=self.metadata)
        else:
            print('!!!!!! Overselection : aborting selection !!!!!!')
            return self
    
    def extract_run(self, dict_coords, show = True):
        '''
        Returns a panda DataFrame and the metadata dict of a SIMA run, according to the dict_coords {model: mmmmm , condition:ccccc, analysis: aaaaa} triplet.
        '''
        run = self.df.sel(dict_coords)
        var_dict_coords = {
            'model':dict_coords['model'],
            'condition':dict_coords['condition']
        }
        var = self.varying_metadata.sel(var_dict_coords)
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

    def skip_transient(self, T_trans):
        '''
        Returns a new class instance whose timeseries datas where chopped to match t > T_trans.
        '''
        time = self.df.coords['time'].values
        t_idx = time>T_trans
        time = time[t_idx]
        new_df = self.df.sel({'time': time})
        return SIMHigh5(new_df, None, source="xarray", name=self.name, metadata=self.metadata)
    
    def select_time_window(self, T_min, T_max):
        '''
        Returns a new class instance whose timeseries datas where chopped to match T_min < t < T_max.
        '''
        time = self.df.coords['time'].values
        T_min,T_max = max(time[0],T_min), min(time[-1],T_max)
        t_idx = np.logical_and(time>=T_min,time<=T_max)
        time = time[t_idx]
        new_df = self.df.sel({'time': time})
        return SIMHigh5(new_df, None, source="xarray", name = self.name, metadata=self.metadata)

    def timeserie(self, sel_dict, output, show=True):
        '''
        Returns tuple (time, serie) from a given output and coordinates.
        '''
        new = self.selection(sel_dict)
        array = new.df[output]
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
    
def merge_simh5(list_of_H5DF):
        '''
        Merges simh5 datasets. Returns a new class instance with merged dataset for the data and the metadata.
        '''
        #TODO have also the time merged and interpolated.
        df_merged = xr.merge([H.df for H in list_of_H5DF], join= 'outer', compat='no_conflicts', combine_attrs='drop')
        metadata_merged = xr.merge([H.metadata for H in list_of_H5DF], join= 'outer', compat='no_conflicts', combine_attrs='drop')
        
        return SIMHigh5(df_merged, None, source='xarray', metadata=metadata_merged, silent=False)
##########################################
# End
##########################################


# Tests
#------------------------------------------

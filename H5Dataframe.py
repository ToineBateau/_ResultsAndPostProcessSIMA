import xarray as xr
import numpy as np
import pandas as pd
import h5py
from analysis_tools import *

# import functions
#------------------------------------------


def strip_H5_to_dataset(every_analysis_output_dict, end_list, last_lvl, lvl_h5_dataset):
    keys = lvl_h5_dataset.keys()
    if not (({'Dynamic'} & set(keys)) or ({'Static'} & set(keys)) or ({'Eigenvalue'} & set(keys))):
        last_lvl = last_lvl[1:] + [None]
        res = []
        for key in keys:
            if key != "Variables":
                last_lvl[-1] = key
                res.append(strip_H5_to_dataset(every_analysis_output_dict, end_list, last_lvl, lvl_h5_dataset[key]))
        if len(res) > 0:
            end_list.extend(res)
        return None
    else:
        var_n_an_keys = list(lvl_h5_dataset.keys())
        metadata = lvl_h5_dataset["Variables"]
        var_n_an_keys.remove('Variables')
        ds_list = []
        dict_vars = {}
        dict_coords = {}
        dict_coords['model'] = [last_lvl[0]]
        dict_coords['condition'] = [last_lvl[-1]]
        dict_coords['analysis'] = []
        times = []
        for analysis in var_n_an_keys:
            dict_coords['analysis'] += [analysis]
            for path, output in every_analysis_output_dict.items():
                try:
                    test = lvl_h5_dataset[analysis][path]
                except :
                    print("No ", output, " found for condition ", last_lvl[-1], 'on model ', last_lvl[0])
                else : 
                    series = lvl_h5_dataset[analysis][path]
                    to = series.attrs['start'][()]
                    dt = series.attrs['delta'][()]
                    N = len(series)
                    te = to + (N-1)*dt
                    time = np.arange(to,te+dt,dt)
                    ds = xr.DataArray(
                        data = [[[series]]],
                        coords = {
                            'model':('model', [last_lvl[0]]),
                            'condition':('condition',[last_lvl[-1]]),
                            'analysis':('analysis', [analysis]),
                            'time':('time', time)
                        },
                        dims = ['model', 'condition', 'analysis', 'time'],
                        attrs = metadata
                    )
                    dict_vars[output] = ds
                    dict_coords['time'] = time
                    times.append('time')

        dset = xr.Dataset(
            data_vars = dict_vars,
            coords = dict_coords
        )
        return(dset)

def dataset_from_h5(h5_file, keys_dict):
    end_list = []
    h5_dataset = h5py.File(h5_file)
    lvls = [list(h5_dataset.keys())[0]]*3
    strip_H5_to_dataset(keys_dict,end_list, lvls, h5_dataset)
    final = end_list[0]
    for ds in end_list[1:]:
        final = final.combine_first(ds)
    print('==============================\n SUCCESS IMPORTING DATASET ')
    print('------------------------------\n THIS IS THE IMPORTED DATASET')
    print(final)
    print('-------------------------------\n SUCCESS IMPORTING AND MERGING DATASET \n-------------------------------')
    return final

##########################################
# Begin
##########################################


class H5Dataframe():
    def __init__(self, h5_data, keys_dict, source = "file", name="NewDataset"):
        self.dims = ['model', 'condition', 'analysis', 'time']
        if source == "file":
            self.df = dataset_from_h5(h5_data, keys_dict)
        elif source == "H5DF":
            self.df = h5_data.df
        elif source == "XRDS":
            self.df = h5_data
        self.name = name

    def selection(self, model = [], condition = [], analysis = [], output = [], time = []):
        return self.df.sel(
            {
                'model':model,
                'condition':condition,
                'analysis':analysis,
                'time':time
            }
        )
    
    def skip_transient(self, T_trans):
        time = self.df.coords['time'].values
        t_idx = time>T_trans
        time = time[t_idx]
        new_df = self.df.sel({'time': time})
        print(new_df)
        return H5Dataframe(new_df, None, source="XRDS")

    def timeserie(self, dict_coords, output):
        array = self.df.sel(dict_coords)[output]
        serie = array.values
        time = array.coords['time'].values
        return time, serie
    
    def comparePlot_timeseries(self, ax, dim_in_legend,  dict_other_fixed_coord, output, legend_list=[], color_list=[]):
        if len(legend_list) == 0:
            l = []
        for i,coord in enumerate(self.df.coords[dim_in_legend].values):
            dict_coords = {dim_in_legend:coord}
            dict_coords.update(dict_other_fixed_coord)
            time, serie = self.timeserie(dict_coords, output)
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

import matplotlib.pyplot as plt
filename = r'.\_ResultsH5\Baseline_\Baseline_KaimalTurbulence_Results.h5'
keys_dict = {
    'platform//Global total position//XGtranslationTotalmotion':'Surge [m]',
    'platform//Global total position//YGtranslationTotalmotion':'Sway [m]',
    'platform//Global total position//ZGtranslationTotalmotion':'Heave [m]',
    'platform//Global total position//XLrotationTotalmotion':'Roll [deg]',
}
HDF5 = H5Dataframe(filename, keys_dict)
print(HDF5.df.sel({
    'analysis':'Dynamic',
    'model':'INO_OptiFLEX22MW_Baseline'
})['Surge [m]'].values[0])

dict_coords = {
    'analysis':'Dynamic',
    'model':'INO_OptiFLEX22MW_Baseline'
}
print('IF WE SELECT : \n--------------------------')
print(HDF5.df.sel(dict_coords))
fig, ax = plt.subplots(2,2, figsize=(8, 4), dpi=100)
HDF5.comparePlot_timeseries(
    ax[0,0],
    'condition',
    dict_coords,
    'Surge [m]'
)

print('SKIP TRANSIENT THING\n-----------------------------\n----------------------------')
H2 = HDF5.skip_transient(900)
print(H2.df)
H2.comparePlot_timeseries(
    ax[0,1],
    'condition',
    dict_coords,
    'Surge [m]'
)

# plt.plot(HDF5.df.sel({
#     'analysis':'Dynamic',
#     'output':'Surge [m]',
#     'model':'INO_OptiFLEX22MW_Baseline'
# })['Surge [m]'].values[0])

# print(HDF5.df.coords['condition'].values)
fig.legend()
plt.show()
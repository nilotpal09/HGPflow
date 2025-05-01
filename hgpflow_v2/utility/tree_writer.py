import uproot
from tqdm import tqdm
import numpy as np
import awkward as ak

class TreeWriter:
    def __init__(self, path, tree_name, chunk_size, dtype_to_32=False):
        self.f = uproot.recreate(path)
        self.tree_name = tree_name

        self.dtype_to_32 = dtype_to_32
        self.fill_one_event_counter = 0
        self.chunk_size = chunk_size
        self.data = {}


    def reset_chunk(self):
        self.data = {}
        self.fill_one_event_counter = 0     


    def type_64_to_32(self, data):
        if np.issubdtype(data[0].dtype, np.integer):
            return ak.values_astype(ak.Array(data), 'int32')
        elif np.issubdtype(data[0].dtype, np.floating):
            return ak.values_astype(ak.Array(data), 'float32')
        elif np.issubdtype(data[0].dtype, np.bool_):
            return ak.values_astype(ak.Array(data), 'bool')
        else:
            raise ValueError(f'Unsupported data type: {data[0].dtype}')


    def write(self):
        if self.data == {}:
            return
        
        if self.dtype_to_32:
            for k, v in self.data.items():
                if type(v) == dict:
                    self.data[k] = ak.zip(v)
                    for subkey, subvalue in v.items():
                        self.data[k][subkey] = self.type_64_to_32(subvalue)
                elif type(v[0]) == np.ndarray:
                    self.data[k] = self.type_64_to_32(v)
                else:
                    self.data[k] = self.type_64_to_32(np.array(v))

        # first time
        if self.tree_name not in self.f:
            self.f[self.tree_name] = self.data
        else:
            self.f[self.tree_name].extend(self.data)



    def fill_one_event(self, data_one_event):

        # create if not exist
        if self.data == {}:
            for k, v in data_one_event.items():
                if type(v) == dict:
                    self.data[k] = {}
                    for kk, vv in v.items():
                        self.data[k][kk] = []
                elif type(v) == np.ndarray: # list
                    self.data[k] = []
                elif type(v) in [int, float, np.int32, np.int64]: # scalar
                    self.data[k] = []
                else:
                    raise ValueError(f'Unsupported data type: ({k}) {type(v)}')

        # fill
        for k, v in data_one_event.items():
            if type(v) == dict:
                for kk, vv in v.items():
                    self.data[k][kk].append(vv)
            else: # same treatment for list and scalar
                self.data[k].append(v)
        self.fill_one_event_counter += 1

        # write if chunk is full
        if self.fill_one_event_counter == self.chunk_size:
            self.write()
            self.reset_chunk()

    
    def close(self):
        self.f.close()


    def write_dict_in_chunk(self, data_dict, desc=""):
        k0 = list(data_dict.keys())[0]
        if type(data_dict[k0]) == dict:
            n_events = len(data_dict[k0][list(data_dict[k0].keys())[0]])
        else:
            n_events = len(data_dict[k0])

        entry_starts = np.arange(0, n_events, self.chunk_size)
        if entry_starts[-1] == n_events:
            entry_starts = entry_starts[:-1]
        entry_stops = np.append(entry_starts[1:], n_events)

        for start, stop in tqdm(zip(entry_starts, entry_stops), total=len(entry_starts), desc=desc):
            self.data = {}
            for k, v in data_dict.items():
                if type(v) == dict:
                    tmp_dict = {}
                    for kk, vv in v.items():
                        tmp_dict[kk] = vv[start:stop]
                    self.data[k] = ak.zip(tmp_dict)
                else:
                    self.data[k] = ak.Array(v[start:stop])

            # HACK: will find an elegant solution later
            if 'truth_inc_shape' in self.data:
                self.data['truth_inc_shape'] = ak.from_iter(self.data['truth_inc_shape'])
                self.data['pred_inc_shape'] = ak.from_iter(self.data['pred_inc_shape'])
            self.write()

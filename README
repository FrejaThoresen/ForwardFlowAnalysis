
# Import

```
export PYTHONPATH=$PATH_TO_FOLDER/helpers:$PYTHONPATH
```

# Example

```python
input_file = $PATH_TO_FILE
directory = $directory_in_path
eta_bins = 48
cent_bins = 80
cent_max = 80
vertex_bins = 10
samples = 10
n_tot  = 3

boot = 500 # no. bootstrap simulations
m = 2 # 2-particle cumulant

flow_object = flow.Flow(input_file, directory, eta_bins, cent_bins, cent_max, vertex_bins, samples, n_tot)
flow_object.read_cumulant_m(m)
flow_object.boot_vnm(boot,m)
```

# TK data sample

In the first step of finding a skills taxonomy from job adverts we take a sample of the job advert data.

This can be done by running:

```
python skills_taxonomy_v2/pipeline/tk_data_analysis/get_tk_sample.py --config_path skills_taxonomy_v2/config/tk_data_sample/2021.10.25.yaml
```

## `2021.10.25.yaml` config file

This samples 5,000,000 job adverts randomly from all the TextKernel files. This sample was further reduced to 4,312,285 job adverts since some of the sample included job adverts which don't have the full text field available.

The output is a dict of each TextKernel file name and a list of the job ids within it which are included in the sample. e.g. `{"historical/...0.json": ['6001f8701aeb4072a8eb0cca85535208', ...]}`. This then provides an easy way to open the original file and get the text from each of the job adverts included in the sample.

This output is saved in `outputs/tk_sample_data/sample_file_locations.json`.

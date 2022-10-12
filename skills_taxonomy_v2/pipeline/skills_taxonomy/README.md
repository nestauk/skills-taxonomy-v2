# Skills Taxonomy

The aim of this pipeline is to build the taxonomy from skills extracted from TextKernel job adverts. There are 2 steps:

1. Build the taxonomy (`build_taxonomy.py`)
2. Output a user friendly version of the taxonomy (`output_taxonomy.py`)

The parameters for all these steps can be found in the config path directory `skills_taxonomy_v2/config/skills_taxonomy/`.

<img src="../../../outputs/reports/figures/Jan 2022/hierarchy_overview.jpg" width="1000">

The latest config file is [`2022.01.21.yaml`](https://github.com/nestauk/skills-taxonomy-v2/blob/7059e46116daf93051347557c7bb69e7e3de64ab/skills_taxonomy_v2/config/skills_taxonomy/2022.01.21.yaml).

## 1. Build the taxonomy

This is run by:
```
python -i skills_taxonomy_v2/pipeline/skills_taxonomy/build_taxonomy.py --config_path 'skills_taxonomy_v2/config/skills_taxonomy/2022.01.21.yaml'
```

Outputs:
- A dictionary of each skill with what part of the hierarchy it is in - `outputs/skills_taxonomy/2022.01.21_skills_hierarchy.json`
- A nested dictionary of each skill group with the skill groups it contains - `outputs/skills_taxonomy/2022.01.21_hierarchy_structure.json`

## 2. Output the taxonomy

Rather than output a json of the hierarchy with numerical keys, this switches the keys to the skill group names. It makes the json output a little bit more user-friendly as a means to interrogate the hierarchy.

Run by:
```
python -i skills_taxonomy_v2/pipeline/skills_taxonomy/output_taxonomy.py --config_path 'skills_taxonomy_v2/config/skills_taxonomy/2022.01.21.yaml'
```

Outputs:
- `outputs/skills_taxonomy/2022.01.21_hierarchy_structure_named.json`

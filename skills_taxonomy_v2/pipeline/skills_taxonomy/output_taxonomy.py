"""
Outputs a slightly more user friendly JSON file for the skills taxonomy.
Uses the names of skill groups and skill rather than numerical identifiers.
"""

import json
import re
from argparse import ArgumentParser
import logging
import yaml

import boto3

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    get_output_config_stamped,
)

logger = logging.getLogger(__name__)

s3 = boto3.resource("s3")


def rename_key_dict(dictionary, new_key):
    """
    If a name is in a dict then give it an updated one
    e.g. {'name': 123, 'key': 23, 'key (2)': 45, 'key (3)': 77}
    """
    if new_key in dictionary:
        # e.g. if dictionary.keys() = 'key', 'key (1)', 'key (2)', 'keyword', 'word (1)'
        # multi_keys = 'key (1)', 'key (2)', 'word (1)'
        # num_previous_times = 3
        regexp = re.compile(r"\((\d+)\)")
        multi_keys = [name for name in dictionary.keys() if regexp.search(name)]
        num_previous_times = len([name for name in multi_keys if new_key in name]) + 1
        renamed_key = new_key + f" ({num_previous_times})"
        return renamed_key
    else:
        return new_key


def parse_arguments(parser):

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/skills_taxonomy/2021.09.06.yaml",
    )

    return parser.parse_args()


if __name__ == "__main__":

    parser = ArgumentParser()
    args = parse_arguments(parser)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    FLOW_ID = "build_taxonomy_flow"

    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    hier_structure_file = get_output_config_stamped(
        args.config_path, params["output_dir"], "hierarchy_structure.json"
    )

    original_hierarchy = load_s3_data(s3, BUCKET_NAME, hier_structure_file)

    # Manual level A names
    with open("skills_taxonomy_v2/utils/2021.09.06_level_a_rename_dict.json", "r") as f:
        level_a_rename_dict = json.load(f)

    hierarchy = {}
    for level_a_num, original_level_a in original_hierarchy.items():
        level_b = {}
        for level_b_num, original_level_b in original_level_a["Level B"].items():
            level_c = {}
            for level_c_num, original_level_c in original_level_b["Level C"].items():
                level_d = {}
                for level_d_num, original_level_d in original_level_c[
                    "Level D"
                ].items():
                    skills = []
                    for skill_num, original_skills in original_level_d[
                        "Skills"
                    ].items():
                        skills.append(skill_num)
                    level_d[original_level_d["Name"]] = skills
                if len(level_d) != len(original_level_c["Level D"]):
                    print(
                        f"Not all level D names in this level C {level_c_num} are unique"
                    )
                level_c[rename_key_dict(level_c, original_level_c["Name"])] = level_d
            if len(level_c) != len(original_level_b["Level C"]):
                print(f"Not all level C names in this level B {level_b_num} are unique")
            level_b[rename_key_dict(level_b, original_level_b["Name"])] = level_c
        if len(level_b) != len(original_level_a["Level B"]):
            print(f"Not all level B names in this level A {level_a_num} are unique")
        hierarchy[
            rename_key_dict(hierarchy, level_a_rename_dict[level_a_num])
        ] = level_b

    new_hier_structure_file = get_output_config_stamped(
        args.config_path, params["output_dir"], "hierarchy_structure_named.json"
    )
    save_to_s3(s3, BUCKET_NAME, hierarchy, new_hier_structure_file)

    logger.info(f"Saved to {new_hier_structure_file}")

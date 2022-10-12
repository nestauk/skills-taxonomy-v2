"""
Outputs a slightly more user friendly JSON file for the skills taxonomy.
Uses the names of skill groups and skill rather than numerical identifiers.
"""

import json
import re
from argparse import ArgumentParser
import logging
import yaml
from collections import defaultdict, Counter

import boto3
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    get_output_config_stamped,
)
from skills_taxonomy_v2.pipeline.skills_taxonomy.build_taxonomy import (
    load_manual_level_dict,
)
from skills_taxonomy_v2.pipeline.skills_taxonomy.build_taxonomy_utils import (
    get_top_tf_idf_words,
)


logger = logging.getLogger(__name__)

s3 = boto3.resource("s3")


def get_new_name(skill_examples, skill_name, skill_num_list):
    skill_texts = [" ".join(skill_examples[s_i]) for s_i in skill_num_list]

    vectorizer = TfidfVectorizer(stop_words="english")
    vect = vectorizer.fit_transform(skill_texts)

    feature_names = np.array(vectorizer.get_feature_names())

    level_names = {
        level_num: skill_name
        + " - "
        + " ".join(get_top_tf_idf_words(doc_vec, feature_names, top_n=2))
        for level_num, doc_vec in zip(skill_num_list, vect)
    }
    return level_names


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
        default="skills_taxonomy_v2/config/skills_taxonomy/2022.01.21.yaml",
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
    if params["level_a_manual_clusters_path"]:
        level_a_rename_dict = load_manual_level_dict(
            params["level_a_manual_clusters_path"]
        )
        level_a_rename_dict = {k: v["Name"] for k, v in level_a_rename_dict.items()}
    else:
        print("No level A manual cluster dictionary given, defaulting to the 2021/12/20 dictionary")
        level_a_rename_dict = load_manual_level_dict(
            "skills_taxonomy_v2/utils/2021.12.20_level_a_rename_dict.json"
        )

    new_hierarchy = {}
    for level_a_num, original_level_a in original_hierarchy.items():
        level_b = {}
        for level_b_num, original_level_b in original_level_a["Level B"].items():
            level_c = {}
            for level_c_num, original_level_c in original_level_b["Level C"].items():
                if "Skills" in original_level_c:
                    level_c[rename_key_dict(level_c, original_level_c["Name"])] = {
                        n: v["Skill name"]
                        for n, v in original_level_c["Skills"].items()
                    }
                else:
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
                    level_c[
                        rename_key_dict(level_c, original_level_c["Name"])
                    ] = level_d
            if len(level_c) != len(original_level_b["Level C"]):
                print(f"Not all level C names in this level B {level_b_num} are unique")
            level_b[rename_key_dict(level_b, original_level_b["Name"])] = level_c
        if len(level_b) != len(original_level_a["Level B"]):
            print(f"Not all level B names in this level A {level_a_num} are unique")
        new_hierarchy[
            rename_key_dict(new_hierarchy, level_a_rename_dict[level_a_num])
        ] = level_b

    # Rename skills:
    # 1. For the skill names the same within a level C group, post-fix a TFIDF couple of words
    # 2. For skill names used in other level C groups after this, post-fix with the level C name in brackets
    # 3. For still the same skill name within level C (with tfidf) add a 1,2,3..

    skills_hierarchy_file = get_output_config_stamped(
        args.config_path, params["output_dir"], "skills_hierarchy.json"
    )

    skill_data = load_s3_data(s3, BUCKET_NAME, skills_hierarchy_file)

    # Get some sentence examples for each skill from the original hierarchy
    skill_examples = {}
    for k1, v1 in original_hierarchy.items():
        for k2, v2 in v1["Level B"].items():
            for k3, v3 in v2["Level C"].items():
                for s_i, skill in v3["Skills"].items():
                    skill_examples[s_i] = skill["Example sentences with skill in"]

    new_skill_names = {}
    for _, leva_info in new_hierarchy.items():
        for _, levb_info in leva_info.items():
            for _, level_c_skills in levb_info.items():

                # Group by skill name
                level_c_skills_mapper = defaultdict(list)
                for skill_num, skill_name in level_c_skills.items():
                    level_c_skills_mapper[skill_name].append(skill_num)

                # Get new names for duplicates
                for skill_name, skill_num_list in level_c_skills_mapper.items():
                    if len(skill_num_list) > 1:
                        level_names = get_new_name(
                            skill_examples, skill_name, skill_num_list
                        )
                        for k, v in level_names.items():
                            new_skill_names[k] = v
                    else:
                        new_skill_names[skill_num_list[0]] = skill_name

    # For multi-level C names (i.e. same name different level C) pre-fix skill name with level C name
    between_dup_skill_names = [
        k for k, v in Counter(list(new_skill_names.values())).items() if v > 1
    ]

    new_dedup_skill_names = {}
    for skill_num, new_skill_name in new_skill_names.items():
        if new_skill_name in between_dup_skill_names:
            new_dedup_skill_names[
                skill_num
            ] = f"{new_skill_name} ({skill_data[skill_num]['Hierarchy level C name']})"
        else:
            new_dedup_skill_names[skill_num] = new_skill_name

    final_dedup_skill_names = {}
    for _, leva_info in new_hierarchy.items():
        for _, levb_info in leva_info.items():
            for _, level_c_skills in levb_info.items():

                # Group by skill name
                level_c_skills_mapper = defaultdict(list)
                for skill_num, _ in level_c_skills.items():
                    skill_name = new_dedup_skill_names[skill_num]
                    level_c_skills_mapper[skill_name].append(skill_num)

                # Get new names for duplicates
                for skill_name, skill_num_list in level_c_skills_mapper.items():
                    if len(skill_num_list) > 1:
                        i = 0
                        for skill_num in skill_num_list:
                            final_dedup_skill_names[skill_num] = (
                                skill_name + " " + str(i)
                            )
                            i += 1
                    else:
                        final_dedup_skill_names[skill_num_list[0]] = skill_name

    logger.info(
        f"New skill names have {len(set(final_dedup_skill_names.values()))} unique names"
    )

    # Add new names to output files and save

    skill_data_new = {}
    for skill_num, skill_info in skill_data.items():
        skill_info["Skill name"] = final_dedup_skill_names[skill_num]
        skill_data_new[skill_num] = skill_info

    new_skills_hierarchy_file = get_output_config_stamped(
        args.config_path, params["output_dir"], "skills_hierarchy_named.json"
    )
    save_to_s3(s3, BUCKET_NAME, skill_data_new, new_skills_hierarchy_file)

    new_new_hierarchy = {}
    for leva_name, leva_info in new_hierarchy.items():
        lev_b_new_hier = {}
        for levb_name, levb_info in leva_info.items():
            lev_c_new_hier = {}
            for levc_name, level_c_skills in levb_info.items():
                skills_new_hier = {}
                for skill_num in level_c_skills.keys():
                    skills_new_hier[skill_num] = final_dedup_skill_names[skill_num]
                lev_c_new_hier[levc_name] = skills_new_hier
            lev_b_new_hier[levb_name] = lev_c_new_hier
        new_new_hierarchy[leva_name] = lev_b_new_hier

    new_hier_structure_file = get_output_config_stamped(
        args.config_path, params["output_dir"], "hierarchy_structure_named.json"
    )
    save_to_s3(s3, BUCKET_NAME, new_new_hierarchy, new_hier_structure_file)

    logger.info(f"Saved to {new_hier_structure_file}")

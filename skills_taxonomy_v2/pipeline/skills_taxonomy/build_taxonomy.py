from argparse import ArgumentParser
import logging
import yaml

import pandas as pd
from tqdm import tqdm
import boto3

from skills_taxonomy_v2.getters.s3_data import load_s3_data, get_s3_data_paths, save_to_s3
from skills_taxonomy_v2 import BUCKET_NAME
from skills_taxonomy_v2.pipeline.skills_taxonomy.build_taxonomy_utils import (
    get_many_clusters,
    get_consensus_clusters_mappings,
    get_top_tf_idf_words,
    get_level_names,
    get_new_level_consensus,
    get_new_level)
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    get_output_config_stamped,
    )

logger = logging.getLogger(__name__)

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

    s3 = boto3.resource("s3")

    sentence_embs = load_s3_data(s3, BUCKET_NAME, params["clustered_sentences_path"])
    sentence_embs = pd.DataFrame(sentence_embs)
    # Remove not clustered sentences
    sentence_embs = sentence_embs[sentence_embs['Cluster number']!=-1]

    skills_data = load_s3_data(s3, BUCKET_NAME, params["skills_data_path"])

    logger.info("Getting lowest level hierarchy ...")
    level_c_cluster_mapper = get_new_level(
        sentence_embs,
        previous_level_col='Cluster number',
        k_means_n=params["level_c_n"],
        k_means_max_iter=params["k_means_max_iter"])
    sentence_embs['Level C'] = sentence_embs['Cluster number'].apply(lambda x: level_c_cluster_mapper[x])
    logger.info(f"Lowest level hierarchy has {sentence_embs['Level C'].nunique()} sections")

    logger.info("Getting mid level hierarchy ...")
    if params["check_low_siloutte_b"]:
        logger.info("Points with low siloutte scores are put in their own cluster")
    level_b_cluster_mapper = get_new_level(
        sentence_embs,
        previous_level_col='Level C',
        k_means_n=params["level_b_n"],
        k_means_max_iter=params["k_means_max_iter"],
        check_low_siloutte=params["check_low_siloutte_b"],
        silhouette_threshold=params["silhouette_threshold"])
    sentence_embs['Level B'] = sentence_embs['Level C'].apply(lambda x: level_b_cluster_mapper[x])
    logger.info(f"Mid level hierarchy has {sentence_embs['Level B'].nunique()} sections")

    logger.info("Getting top level hierarchy ...")
    if params["use_level_a_consensus"]:
        logger.info("... using consensus clustering")
        level_a_cluster_mapper = get_new_level_consensus(
            sentence_embs,
            previous_level_col='Level B',
            k_means_n=params["level_a_n"],
            numclust_its=params["level_a_consensus_numclust_its"]
            )
        sentence_embs['Level A'] = sentence_embs['Level B'].apply(lambda x: level_a_cluster_mapper[x])
    else:
        level_a_cluster_mapper = get_new_level(
            sentence_embs,
            previous_level_col='Level B',
            k_means_n=params["level_a_n"],
            k_means_max_iter=params["k_means_max_iter"]
            )
        sentence_embs['Level A'] = sentence_embs['Level B'].apply(lambda x: level_a_cluster_mapper[x])
    logger.info(f"Top level hierarchy has {sentence_embs['Level A'].nunique()} sections")

    # Level D is just a merging of level C skills which were given the same name (i.e. no clustering)
    # If there are skills with the same name in level C then group these
    sentence_embs['Skill name'] = sentence_embs['Cluster number'].apply(lambda x: skills_data[str(x)]['Skills name'])
    level_c_name_mapper = {}
    for level_c_num, level_c_data in sentence_embs.groupby('Level C'):
        level_c_skill_names = level_c_data['Skill name'].unique().tolist()
        level_c_name_mapper[level_c_num] = {level_c_skill_name:i for i, level_c_skill_name in enumerate(level_c_skill_names)}

    def get_level_c_merged_names(skill):
        return level_c_name_mapper[skill['Level C']][skill['Skill name']]

    sentence_embs['Level D'] = sentence_embs.apply(get_level_c_merged_names, axis=1)
   
    # Level names
    level_a_names = get_level_names(sentence_embs, 'Level A', top_n=params["level_names_tfidif_n"])
    level_b_names = get_level_names(sentence_embs, 'Level B', top_n=params["level_names_tfidif_n"])
    level_c_names = get_level_names(sentence_embs, 'Level C', top_n=params["level_names_tfidif_n"])

    logger.info("Creating and saving dictionary of hierarchical information per skill ...")
    # Dict of hierarchy information per skill
    # {skill_num: {hierarchy info for this skill}}

    skill_hierarchy = {}
    for skill_num, skill_info in skills_data.items():
        skill_num = int(skill_num)
        if skill_num != -1:
            hier_info = {}
            level_c = level_c_cluster_mapper[skill_num]
            level_b = level_b_cluster_mapper[level_c]
            level_a = level_a_cluster_mapper[level_b]
            hier_info['Skill name'] = skill_info['Skills name']
            hier_info['Hierarchy level A'] = level_a
            hier_info['Hierarchy level B'] = level_b
            hier_info['Hierarchy level C'] = level_c
            hier_info['Hierarchy level D'] = level_c_name_mapper[level_c][skill_info['Skills name']]
            hier_info['Hierarchy level A name'] = level_a_names[level_a]
            hier_info['Hierarchy level B name'] = level_b_names[level_b]
            hier_info['Hierarchy level C name'] = level_c_names[level_c]
            hier_info['Hierarchy ID'] = f"{level_a}-{level_b}-{level_c}"
            hier_info['Number of sentences that created skill'] = len(skill_info['Texts'])
            skill_hierarchy[skill_num] = hier_info

    # Save json
    skill_hierarchy_file = get_output_config_stamped(
        args.config_path, params["output_dir"], "skills_hierarchy.json"
    )
    save_to_s3(s3, BUCKET_NAME, skill_hierarchy, skill_hierarchy_file)
    logger.info(f"Saved to {skill_hierarchy_file}")


    logger.info("Creating and saving dictionary of hierarchical information per level ...")
    # Dict of hierarchy information per level
    # {level_a_num: {level_b_info: {level_c_info}}}

    hier_structure = {}
    for level_a_num, level_a_num_data in sentence_embs.groupby('Level A'):
        level_b_structure = {}
        for level_b_num, level_b_num_data in level_a_num_data.groupby('Level B'):
            level_c_structure = {}
            for level_c_num, level_c_num_data in level_b_num_data.groupby('Level C'):
                level_d_structure = {}
                for level_d_num, level_d_num_data in level_c_num_data.groupby('Level D'):
                    skill_nums = level_d_num_data['Cluster number'].unique().tolist()
                    # The name at this level is the skill names all these level Ds are grouped on
                    level_d_structure[level_d_num] = {
                        'Name': skill_hierarchy[skill_nums[0]]['Skill name'],
                        'Number of skills': len(skill_nums), 
                        'Skills': {k: {
                            'Skill name': skill_hierarchy[k]['Skill name'],
                            'Number of sentences that created skill': skill_hierarchy[k]['Number of sentences that created skill'],
                            } for k in skill_nums}
                    }
                skill_nums_c = level_c_num_data['Cluster number'].unique().tolist()
                level_c_structure[level_c_num] = {
                    'Name': level_c_names[level_c_num],
                    'Number of skills': len(skill_nums_c), 
                    'Level D': level_d_structure
                }
            skill_nums_b = level_b_num_data['Cluster number'].unique().tolist()
            level_b_structure[level_b_num] = {
                'Name': level_b_names[level_b_num],
                'Number of skills': len(skill_nums_b), 
                'Level C': level_c_structure
            }
        skill_nums_a = level_a_num_data['Cluster number'].unique().tolist()
        hier_structure[level_a_num] = {
            'Name': level_a_names[level_a_num],
            'Number of skills': len(skill_nums_a), 
            'Level B': level_b_structure
        }

    # Save json
    hier_structure_file = get_output_config_stamped(
        args.config_path, params["output_dir"], "hierarchy_structure.json"
    )
    save_to_s3(s3, BUCKET_NAME, hier_structure, hier_structure_file)

    logger.info(f"Saved to {hier_structure_file}")

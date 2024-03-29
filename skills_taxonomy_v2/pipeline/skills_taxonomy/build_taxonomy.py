from argparse import ArgumentParser
import logging
import yaml
import json

import pandas as pd
from tqdm import tqdm
import boto3

from skills_taxonomy_v2.getters.s3_data import (
    load_s3_data,
    get_s3_data_paths,
    save_to_s3,
)
from skills_taxonomy_v2 import BUCKET_NAME
from skills_taxonomy_v2.pipeline.skills_taxonomy.build_taxonomy_utils import (
    get_many_clusters,
    get_consensus_clusters_mappings,
    get_top_tf_idf_words,
    get_level_names,
    get_new_level_consensus,
    get_new_level,
    amend_level_b_mapper,
    manual_cluster_level,
)
from skills_taxonomy_v2.pipeline.skills_extraction.extract_skills_utils import (
    get_output_config_stamped,
)

logger = logging.getLogger(__name__)


def load_skills_data(s3, skills_data_path):
    return load_s3_data(s3, BUCKET_NAME, skills_data_path)


def load_sentences_data(
    s3, cluster_column_name, clustered_sentences_path, reduced_embeddings_dir=None
):

    # The sentences ID + cluster num
    sentence_embs = load_s3_data(s3, BUCKET_NAME, clustered_sentences_path)
    if "lightweight" in clustered_sentences_path:
        sentence_embs = pd.DataFrame(
            sentence_embs,
            columns=['job id', 'sentence id',  'Cluster number predicted']
            )
    else:
        sentence_embs = pd.DataFrame(sentence_embs)

    # (new) Get the reduced embeddings + sentence texts and the sentence IDs
    if reduced_embeddings_dir:
        reduced_embeddings_paths = get_s3_data_paths(
            s3,
            BUCKET_NAME,
            reduced_embeddings_dir,
            file_types=["*sentences_data_*.json"],
        )

        sentences_data = pd.DataFrame()
        for reduced_embeddings_path in tqdm(reduced_embeddings_paths):
            sentences_data_i = load_s3_data(s3, BUCKET_NAME, reduced_embeddings_path)
            sentences_data = pd.concat([sentences_data, pd.DataFrame(sentences_data_i)])
        sentences_data.reset_index(drop=True, inplace=True)

        # Merge the reduced embeddings + texts with the sentence ID+cluster number
        sentence_embs = pd.merge(
            sentences_data, sentence_embs, how="left", on=["job id", "sentence id"]
        )
        sentence_embs["description"] = sentence_embs["description"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else x
        )

    # Remove not clustered sentences
    sentence_embs = sentence_embs[sentence_embs[cluster_column_name] >= 0]

    return sentence_embs


def load_manual_level_dict(level_a_manual_clusters_path):
    with open(level_a_manual_clusters_path, "r") as f:
        levela_manual = json.load(f)
    return levela_manual


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

    s3 = boto3.resource("s3")

    cluster_column_name = params.get("cluster_column_name", "Cluster name")
    embedding_column_name = params.get("embedding_column_name", "reduced_points_umap")

    # Load skills and sentences data
    skills_data = load_skills_data(s3, params["skills_data_path"])
    sentence_embs = load_sentences_data(
        s3,
        cluster_column_name,
        params["clustered_sentences_path"],
        params["reduced_embeddings_dir"],
    )
    logger.info(
        f"Building hierarchy with {len(sentence_embs)} sentences and {len(skills_data)} skills"
    )

    # Join the skill names to the skills data
    if params.get("skills_names_data_path"):
        skills_names = load_skills_data(s3, params["skills_names_data_path"])
        for skill_num, skill_info in skills_data.items():
            skills_data[skill_num]["Skills name"] = skills_names[skill_num]["Skills name"]

    if params.get("level_a_manual_clusters_path"):
        levela_manual = load_manual_level_dict(params["level_a_manual_clusters_path"])
    else:
        levela_manual = None

    create_level_d = params.get("create_level_d", True)
    skills_data_texts_name = params.get("skills_data_texts_name", "Texts")

    # Build hierarchy

    logger.info("Getting lowest level hierarchy ...")
    level_c_cluster_mapper = get_new_level(
        sentence_embs,
        previous_level_col=cluster_column_name,
        k_means_n=params["level_c_n"],
        k_means_max_iter=params["k_means_max_iter"],
        embedding_column_name=embedding_column_name,
    )
    sentence_embs["Level C"] = sentence_embs[cluster_column_name].apply(
        lambda x: level_c_cluster_mapper[x]
    )
    logger.info(
        f"Lowest level hierarchy has {sentence_embs['Level C'].nunique()} sections"
    )

    logger.info("Getting mid level hierarchy ...")
    if params["check_low_siloutte_b"]:
        logger.info("Points with low siloutte scores are put in their own cluster")
        silhouette_threshold = params["silhouette_threshold"]
    else:
        silhouette_threshold = None
    level_b_cluster_mapper = get_new_level(
        sentence_embs,
        previous_level_col="Level C",
        k_means_n=params["level_b_n"],
        k_means_max_iter=params["k_means_max_iter"],
        check_low_siloutte=params["check_low_siloutte_b"],
        silhouette_threshold=silhouette_threshold,
        embedding_column_name=embedding_column_name,
    )

    if levela_manual:
        logger.info(
            "Using a manually created mapper to regroup level C groups - if data/random seed has changed this mapper may be out of date."
        )
        level_b_cluster_mapper = amend_level_b_mapper(
            level_b_cluster_mapper, levela_manual
        )

    print(
        f"Mid level hierarchy has {len(set(level_b_cluster_mapper.values()))} sections"
    )

    sentence_embs["Level B"] = sentence_embs["Level C"].apply(
        lambda x: level_b_cluster_mapper[x]
    )
    logger.info(
        f"Mid level hierarchy has {sentence_embs['Level B'].nunique()} sections"
    )

    logger.info("Getting top level hierarchy ...")
    if levela_manual:
        logger.info(
            "Using a manually created mapper to group level B groups - if data/random seed has changed this mapper may be out of date."
        )
        level_a_cluster_mapper = manual_cluster_level(
            levela_manual, level_b_cluster_mapper
        )
    else:
        if params.get("use_level_a_consensus"):
            logger.info("... using consensus clustering")
            level_a_cluster_mapper = get_new_level_consensus(
                sentence_embs,
                previous_level_col="Level B",
                k_means_n=params["level_a_n"],
                numclust_its=params["level_a_consensus_numclust_its"],
                embedding_column_name=embedding_column_name,
            )
        else:
            level_a_cluster_mapper = get_new_level(
                sentence_embs,
                previous_level_col="Level B",
                k_means_n=params["level_a_n"],
                k_means_max_iter=params["k_means_max_iter"],
                embedding_column_name=embedding_column_name,
            )
    sentence_embs["Level A"] = sentence_embs["Level B"].apply(
        lambda x: level_a_cluster_mapper[x]
    )

    logger.info(
        f"Top level hierarchy has {sentence_embs['Level A'].nunique()} sections"
    )

    try:
        # If the skill name is given in the skills data then use it
        sentence_embs["Skill name"] = sentence_embs[cluster_column_name].apply(
            lambda x: skills_data[str(x)]["Skills name"]
        )
    except KeyError:
        # Create a skill name using TF-IDF
        skill_tfidf = get_level_names(
            sentence_embs, cluster_column_name, top_n=params["level_names_tfidif_n"]
        )
        sentence_embs["Skill name"] = sentence_embs[cluster_column_name].apply(
            lambda x: skill_tfidf[x]
        )

    if create_level_d:
        # Level D is just a merging of level C skills which were given the same name (i.e. no clustering)
        # If there are skills with the same name in level C then group these
        level_c_name_mapper = {}
        for level_c_num, level_c_data in sentence_embs.groupby("Level C"):
            level_c_skill_names = level_c_data["Skill name"].unique().tolist()
            level_c_name_mapper[level_c_num] = {
                level_c_skill_name: i
                for i, level_c_skill_name in enumerate(level_c_skill_names)
            }

        def get_level_c_merged_names(skill):
            return level_c_name_mapper[skill["Level C"]][skill["Skill name"]]

        sentence_embs["Level D"] = sentence_embs.apply(get_level_c_merged_names, axis=1)

    # Level names
    if levela_manual:
        level_a_names = {int(k):v['Name'] for k, v in levela_manual.items()}
    else:
        level_a_names = get_level_names(
            sentence_embs, "Level A", top_n=params["level_names_tfidif_n"]
        )
    level_b_names = {}
    for level_a_num, level_data in sentence_embs.groupby("Level A"):
        level_b_names.update(get_level_names(
            level_data, "Level B", top_n=params["level_names_tfidif_n"], max_df=1.0
        ))

    level_c_names = {}
    for level_b_num, level_b_data in sentence_embs.groupby("Level B"):
        level_c_names.update(get_level_names(
            level_b_data, "Level C", top_n=params["level_names_tfidif_n"], max_df=1.0
        ))

    logger.info(
        "Creating and saving dictionary of hierarchical information per skill ..."
    )
    # Dict of hierarchy information per skill
    # e.g. {skill_num: {hierarchy info for this skill}, ...}

    skill_hierarchy = {}
    for skill_num, skill_info in skills_data.items():
        skill_num = int(skill_num)
        if skill_num != -1:
            hier_info = {}
            level_c = level_c_cluster_mapper[skill_num]
            level_b = level_b_cluster_mapper[level_c]
            level_a = level_a_cluster_mapper[level_b]
            try:
                hier_info["Skill name"] = skill_info["Skills name"]
            except KeyError:
                hier_info["Skill name"] = skill_tfidf[skill_num]
            hier_info["Hierarchy level A"] = level_a
            hier_info["Hierarchy level B"] = level_b
            hier_info["Hierarchy level C"] = level_c
            try:
                hier_info["Hierarchy level D"] = level_c_name_mapper[level_c][
                    skill_info["Skills name"]
                ]
            except NameError:
                pass
            hier_info["Skill centroid"] = skill_info["Centroid"]
            hier_info["Hierarchy level A name"] = level_a_names[level_a]
            hier_info["Hierarchy level B name"] = level_b_names[level_b]
            hier_info["Hierarchy level C name"] = level_c_names[level_c]
            hier_info["Hierarchy ID"] = f"{level_a}-{level_b}-{level_c}"
            hier_info["Number of sentences that created skill"] = len(
                skill_info[skills_data_texts_name]
            )
            skill_hierarchy[skill_num] = hier_info

    # Save json
    skill_hierarchy_file = get_output_config_stamped(
        args.config_path, params["output_dir"], "skills_hierarchy.json"
    )
    save_to_s3(s3, BUCKET_NAME, skill_hierarchy, skill_hierarchy_file)
    logger.info(f"Saved to {skill_hierarchy_file}")

    logger.info(
        "Creating and saving dictionary of hierarchical information per level ..."
    )
    # Dict of hierarchy information per level
    # e.g. {level_a_num: {level_b_info: {level_c_info}}}

    hier_structure = {}
    for level_a_num, level_a_num_data in sentence_embs.groupby("Level A"):
        level_b_structure = {}
        for level_b_num, level_b_num_data in level_a_num_data.groupby("Level B"):
            level_c_structure = {}
            for level_c_num, level_c_num_data in level_b_num_data.groupby("Level C"):
                if create_level_d:
                    level_d_structure = {}
                    for level_d_num, level_d_num_data in level_c_num_data.groupby(
                        "Level D"
                    ):
                        skill_nums = (
                            level_d_num_data[cluster_column_name].unique().tolist()
                        )
                        # The name at this level is the skill names all these level Ds are grouped on
                        level_d_structure[level_d_num] = {
                            "Name": skill_hierarchy[skill_nums[0]]["Skill name"],
                            "Number of skills": len(skill_nums),
                            "Skills": {
                                k: {
                                    "Skill name": skill_hierarchy[k]["Skill name"],
                                    "Number of sentences that created skill": skill_hierarchy[
                                        k
                                    ][
                                        "Number of sentences that created skill"
                                    ],
                                    "Example sentences with skill in": skills_data[
                                        str(k)
                                    ][skills_data_texts_name][0:10],
                                }
                                for k in skill_nums
                            },
                        }
                    skill_nums_c = (
                        level_c_num_data[cluster_column_name].unique().tolist()
                    )
                    level_c_structure[level_c_num] = {
                        "Name": level_c_names[level_c_num],
                        "Number of skills": len(skill_nums_c),
                        "Level D": level_d_structure,
                    }
                else:
                    skill_nums = level_c_num_data[cluster_column_name].unique().tolist()
                    level_c_structure[level_c_num] = {
                        "Name": level_c_names[level_c_num],
                        "Number of skills": len(skill_nums),
                        "Skills": {
                            k: {
                                "Skill name": skill_hierarchy[k]["Skill name"],
                                "Number of sentences that created skill": skill_hierarchy[
                                    k
                                ][
                                    "Number of sentences that created skill"
                                ],
                                "Example sentences with skill in": skills_data[str(k)][
                                    skills_data_texts_name
                                ][0:10],
                            }
                            for k in skill_nums
                        },
                    }
            skill_nums_b = level_b_num_data[cluster_column_name].unique().tolist()
            level_b_structure[level_b_num] = {
                "Name": level_b_names[level_b_num],
                "Number of skills": len(skill_nums_b),
                "Level C": level_c_structure,
            }
        skill_nums_a = level_a_num_data[cluster_column_name].unique().tolist()
        hier_structure[level_a_num] = {
            "Name": level_a_names[level_a_num],
            "Number of skills": len(skill_nums_a),
            "Level B": level_b_structure,
        }

    # Save json
    hier_structure_file = get_output_config_stamped(
        args.config_path, params["output_dir"], "hierarchy_structure.json"
    )
    save_to_s3(s3, BUCKET_NAME, hier_structure, hier_structure_file)

    logger.info(f"Saved to {hier_structure_file}")

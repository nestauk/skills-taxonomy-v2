# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Comparing the whole TextKernel dataset to the sample from which skills are extracted
#
# Compare the sample of TK job adverts used in the skills with all the TK data.

# %%
# cd ../../..

# %%
from skills_taxonomy_v2.getters.s3_data import load_s3_data, get_s3_data_paths

from collections import Counter, defaultdict

from datetime import datetime
from tqdm import tqdm
import pandas as pd
import boto3
import matplotlib.pyplot as plt

BUCKET_NAME = "skills-taxonomy-v2"
s3 = boto3.resource("s3")

# %%
file_date = '2021.11.05'

# %% [markdown]
# ## Investigate where job id has gone skewed
# - Not expired files from all data
# - Not expired files from 5 million sample (4,312,285 job ids)
# - skill sentences output (3,572,140 job ids)
# - embeddings output (3,572,084 job ids)
# - reduced embeddings output (1,012,869 job ids)
# - extracted skills output (1,012,869 job ids)

# %%
# All TK data no expired no full text
all_tk_notexpired_nofulltext_counts = load_s3_data(
    s3, BUCKET_NAME,
    "outputs/tk_data_analysis_new_method/metadata_date/tk_dates_count_not_expired_full_text.json"
)


# %%
# All TK data but not expired files
all_tk_notexpired_counts = load_s3_data(
    s3, BUCKET_NAME, "outputs/tk_data_analysis_new_method/metadata_date/tk_not_expired_count.json"
)

# %%
# The 5 million sample
sample_dict = load_s3_data(
    s3, BUCKET_NAME, "outputs/tk_sample_data/sample_file_locations.json"
)

# %%
print(sum([len(v) for v in sample_dict.values()]))
sample_dict_not_expired_files = {k:v for k,v in sample_dict.items() if 'jobs_expired' not in k}
job_ids_not_expired_files = set([f for l in sample_dict_not_expired_files.values() for f in l])
len(job_ids_not_expired_files)


# %%
# The job ids which have skill sentences found
all_job_ids_with_skill_sents = load_s3_data(
    s3, BUCKET_NAME,
    "outputs/tk_data_analysis_new_method/job_ids_in_skill_sentences_2021.10.27_textkernel-files.json")
all_job_ids_with_skill_sents = set(all_job_ids_with_skill_sents)
len(all_job_ids_with_skill_sents)


# %%
# The job ids which have sentence embeddings found
all_job_ids_with_embs = load_s3_data(
    s3, BUCKET_NAME,
    "outputs/tk_data_analysis_new_method/job_ids_in_skills_extraction_word_embeddings_data_2021.11.05.json")
all_job_ids_with_embs = set(all_job_ids_with_embs)
len(all_job_ids_with_embs)


# %%
# The job ids which have reduced embeddings found
reduced_embeddings_paths = get_s3_data_paths(
            s3,
            BUCKET_NAME,
            "outputs/skills_extraction/reduced_embeddings/",
            file_types=["*sentences_data_*.json"]
            )

skills_embeds_df = pd.DataFrame()
for reduced_embeddings_path in tqdm(reduced_embeddings_paths):
    sentences_data_i = load_s3_data(
        s3, BUCKET_NAME,
        reduced_embeddings_path
    )
    skills_embeds_df = pd.concat([skills_embeds_df, pd.DataFrame(sentences_data_i)])
skills_embeds_df.reset_index(drop=True, inplace=True)

all_job_ids_with_red_embs = set(list(skills_embeds_df['job id'].unique()))
len(all_job_ids_with_red_embs)

# %%
# The job ids which have skills extracted (inc -2) from
sentence_data = load_s3_data(s3, BUCKET_NAME, "outputs/skills_extraction/extracted_skills/2021.11.05_sentences_skills_data.json")
sentence_data_df = pd.DataFrame(sentence_data)
all_job_ids_with_skill_extracted = set(sentence_data_df['job id'].unique())
len(all_job_ids_with_skill_extracted)


# %% [markdown]
# ## Get all the dates counted

# %%
# All the TK counts (pre-calculated)
all_tk_job_dates_count = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis_new_method/metadata_date/all_tk_date_count.json")


# %%
## The dates for all job ids in the 5 million sample
# sample_job_dates = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis/metadata_date/sample_filtered_2021.11.05.json")

sample_job_dates_from_metadata = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis_new_method/metadata_date/sample_filtered_2021.11.05_from_metadata.json")



# %%
# Each job id's dates found
unique_dates_lists = defaultdict(list)
for job_id, dates_lists in sample_job_dates_from_metadata.items():
    for date_list in dates_lists:
        for date in date_list:
            unique_dates_lists[job_id].append(date)
        
sample_job_dates_dupes = {k:list(set(v)) for k, v in unique_dates_lists.items()}
Counter([len(v) for v in sample_job_dates_dupes.values()])

# %%
# Just one date per job id
sample_job_dates = {}
weird_jobs = {}
for job_id, date_list in unique_dates_lists.items():
    dates = [date for date in list(set(date_list)) if date]
    if len(dates)==0:
        sample_job_dates[job_id] = None
    elif len(dates)==1:
        # Majority of cases
        sample_job_dates[job_id] = dates[0]
    else:
        weird_jobs[job_id] = dates
        sample_job_dates[job_id] = dates[0]

# %%
print(len(weird_jobs)/len(unique_dates_lists))
# Most of the time they are the same year!
print(sum([v[0][0:4]==v[1][0:4] for v in weird_jobs.values()])/len(weird_jobs))
# Most of the time they are the same year-month
print(sum([v[0][0:7]==v[1][0:7] for v in weird_jobs.values()])/len(weird_jobs))

# %% [markdown]
# ### Investigate whether the 2020s sample data has no text

# %%
sample_dict_job_id2filename = {}
already_in = []
for file_name, job_id_list in tqdm(sample_dict.items()):
    for job_id in job_id_list:
        if job_id in sample_dict_job_id2filename:
            already_in.append(job_id)
        sample_dict_job_id2filename[job_id] = file_name
    

# %%
len(already_in)

# %%
sample_202s_file_names = list(set([sample_dict_job_id2filename[j] for j, d in sample_job_dates.items() if (d and d[0:3]=='202')]))


# %%
len(sample_202s_file_names)

# %%
not_expired_202s_sample = [s for s in sample_202s_file_names if "jobs_expired" not in s]
print(len(not_expired_202s_sample))
not_expired_202s_sample[0:3]

# %%
[i for i, f in enumerate(not_expired_202s_sample) if f=='semiannual/2020/2020-10-02/jobs_new.27.jsonl.gz']

# %%
no_text_job_ids = defaultdict(list)
tk_data_path = "inputs/data/textkernel-files/"
for file_name in tqdm(not_expired_202s_sample[5:]):
    orig_data_file = load_s3_data(s3, BUCKET_NAME, tk_data_path+file_name)
    for d in orig_data_file:
        if not d.get('full_text'):
            no_text_job_ids[file_name].append(d.get('job_id'))
    if file_name in no_text_job_ids:
        print(len(no_text_job_ids[file_name]))


# %%
len(no_text_job_ids)

# %% [markdown]
# ### Reformate dates for plotting

# %%
all_tk_job_dates_count_ym = defaultdict(int)
for k, v in tqdm(all_tk_job_dates_count.items()):
    if k!="Not given":
        date = k[0:7]
    else:
        date = k
    all_tk_job_dates_count_ym[date] += v
    

# %%
all_tk_notexpired_counts_ym = defaultdict(int)
for k, v in tqdm(all_tk_notexpired_counts.items()):
    if k!="Not given":
        date = k[0:7]
    else:
        date = k
    all_tk_notexpired_counts_ym[date] += v

# %%
sample_job_dates_count = defaultdict(int)
for k, v in tqdm(sample_job_dates.items()):
    if v:
        date = v[0:7]
        sample_job_dates_count[date] += 1
    else:
        sample_job_dates_count["Not given"] += 1


# %%
def count_dates(job_dates):

    job_dates_count = defaultdict(int)
    for k, v in tqdm(job_dates.items()):
        if v:
            date = v[0:7]
            job_dates_count[date] += 1
        else:
            job_dates_count["Not given"] += 1
    return job_dates_count



# %%
def find_num_dates(count_dict):
    num_dates = {
        int(k.split("-")[0]) + int(k.split("-")[1]) / 12: v
        for k, v in count_dict.items()
        if k != "Not given"
    }
    num_dates[2014] = count_dict["Not given"]
    return num_dates


# %%
num_dates = find_num_dates(all_tk_job_dates_count_ym)
num_dates_sample = find_num_dates(sample_job_dates_count)
num_dates_notexpired = find_num_dates(all_tk_notexpired_counts_ym)

job_dates_sample_not_expired = {k:v for k,v in sample_job_dates.items() if k in job_ids_not_expired_files}
job_dates_skill_sents = {k:v for k,v in sample_job_dates.items() if k in all_job_ids_with_skill_sents}
job_dates_embs = {k:v for k,v in sample_job_dates.items() if k in all_job_ids_with_embs}
job_dates_red_embs = {k:v for k,v in sample_job_dates.items() if k in all_job_ids_with_red_embs}
job_dates_skill_extracted = {k:v for k,v in sample_job_dates.items() if k in all_job_ids_with_skill_extracted}


num_dates_sample_not_expired = find_num_dates(count_dates(job_dates_sample_not_expired))
num_dates_with_skill_sents = find_num_dates(count_dates(job_dates_skill_sents))
num_dates_with_embs = find_num_dates(count_dates(job_dates_embs))
num_dates_with_red_embs = find_num_dates(count_dates(job_dates_red_embs))
num_dates_with_skill_extracted= find_num_dates(count_dates(job_dates_skill_extracted))


# %%
print("All jobs data")
print(sum(all_tk_job_dates_count_ym.values()))
print(sum(num_dates.values()))

print("All jobs data - no expired files")
print(sum(all_tk_notexpired_counts_ym.values()))
print(sum(num_dates_notexpired.values()))

print("Original 5mill job sample")
print(sum(sample_job_dates_count.values()))
print(sum(num_dates_sample.values()))


print("Job ids with skill sentences")
print(len(all_job_ids_with_skill_sents))
print(len(job_dates_skill_sents))
print(sum(num_dates_with_skill_sents.values()))

print("Job ids with reduced embs")
print(len(all_job_ids_with_red_embs))
print(len(job_dates_red_embs))
print(sum(num_dates_with_red_embs.values()))

print("Job ids with skills extracted")
print(len(all_job_ids_with_skill_extracted))
print(len(job_dates_skill_extracted))
print(sum(num_dates_with_skill_extracted.values()))


# %%
def plot_prop_data(dates, no_none, plt, label, color, alpha=0.5):
    if no_none:
        dates = {k:v for k,v in dates.items() if k!=2014}
    plt.bar(
        dates.keys(),
        [v / sum(dates.values()) for v in dates.values()],
        width=0.1,
        alpha=alpha,
        color=color,
        label=label,
    )


# %%
plt.figure(figsize=(10, 4))
no_none=True

plot_prop_data(num_dates, no_none, plt, label="All data", color="blue", alpha=1)
plot_prop_data(num_dates_notexpired, no_none, plt, label="All data - not expired", color="black")
plot_prop_data(num_dates_sample, no_none, plt, label="Sample of data",  color="red")
plot_prop_data(num_dates_sample_not_expired, no_none, plt, label="Sample of data with not expired files", color="black")
plot_prop_data(num_dates_with_skill_sents, no_none, plt, label="Sample of data with skill sentences", color="green")
plot_prop_data(num_dates_with_embs, no_none, plt, label="Sample of data with embs", color="magenta")
plot_prop_data(num_dates_with_red_embs, no_none, plt, label="Sample of data with reduced embs", color="yellow")
plot_prop_data(num_dates_with_skill_extracted, no_none, plt, label="Sample of data with skill sents", color="purple")

plt.legend()
plt.xlabel("Date of job advert (2014 = no date given)")
plt.ylabel("Proportion")

# %% [markdown]
# ## Add the extra sentences which were deduplicated in the `reduce_embeddings.py` step
#
# The plot shows the change in distribution happens between the embeddings and the reduced embeddings steps. On investigation I see this is because we don't bother including the duplicated sentences (masked) in the data used for reduction (and clustering).
#
# So, in `get_duplicate_sentence_skills.py` we create a dictionary of the words unique id (different from the sentence id since that was creating using the original sentence not the masked one). This looks like:
#
# `{word_id: [[job_id, sent_id], [job_id, sent_id]], word_id: [[job_id, sent_id]], ...}`
#
# Now we need to include these duplicate job ids in our sample. So 
# 1. For our skill sentences, find the newly created `word_id` that matches the `[job_id, sent_id]`
# 2. Get all the other `job_id`s that have this `word_id`.
# 3. Add these to `sentence_data_df`

# %%
job_id_sents_clusters = defaultdict(list)
for s in tqdm(sentence_data):
    job_id_sents_clusters[s['job id']].append([s['sentence id'], s['Cluster number predicted']])
    

# %%
# words_id_dict = load_s3_data(
#     s3, BUCKET_NAME, "outputs/tk_data_analysis_new_method/2021.11.05_words_id_dict.json"
# )
# len(words_id_dict)

# %%
words_id_list = []
for i in tqdm([0,1]):
    words_id_list_temp = load_s3_data(
        s3, BUCKET_NAME, f"outputs/tk_data_analysis_new_method/2021.11.05_words_id_list_{i}.json"
    )    
    words_id_list += words_id_list_temp

print(len(words_id_list))


# %%
i=6
words_id_list_temp = load_s3_data(
        s3, BUCKET_NAME, f"outputs/tk_data_analysis_new_method/2021.11.05_words_id_list_{i}.json"
    )
words_id_list += words_id_list_temp

# %%
print(len(words_id_list))

# %%
# This is handy for next steps where we filter out -2 clusters
# {word_id: cluster num}
word_id2cluster_dict = {}
for words_id, job_id, sentence_id in tqdm(words_id_list):
    possible_sents = job_id_sents_clusters[job_id]
    if possible_sents!=[]:
        clust_nums = [p_clust for p_sent_id, p_clust in possible_sents if p_sent_id==sentence_id]
        if clust_nums!=[]:
            word_id2cluster_dict[words_id] = clust_nums[0]

    

# %%
len(word_id2cluster_dict)

# %%
words_id_list_df = pd.DataFrame(words_id_list, columns=['words_id', 'job id','sentence id'])
len(words_id_list_df)

# %%
# Concat with the all the word id data
print(len(sentence_data_df))
sentence_data_df_enhanced_all = pd.concat([sentence_data_df, words_id_list_df])
print(len(sentence_data_df_enhanced_all))

# Remove duplicates (which will exist from this process)
sentence_data_df_enhanced_all.drop_duplicates(subset=['job id', "sentence id"], inplace=True)

sentence_data_df_enhanced_all.reset_index(inplace=True)
print(len(sentence_data_df_enhanced_all))

# %%
# # Add the word id's to existing sentences
# print(len(sentence_data_df))
# sentence_data_df_enhanced = pd.merge(sentence_data_df, words_id_list_df, how="left", on = ['job id', "sentence id"])
# print(len(sentence_data_df_enhanced))

# # Create a word id 2 cluster num dict
# word_id2cluster_dict = sentence_data_df_enhanced.set_index('words_id').to_dict()["Cluster number predicted"]


# %%
# # Concat with the all the word id data
# sentence_data_df_enhanced_all = pd.concat([sentence_data_df_enhanced, words_id_list_df])
# print(len(sentence_data_df_enhanced_all))

# # Remove duplicates (which will exist from this process)
# sentence_data_df_enhanced_all.drop_duplicates(subset=['job id', "sentence id"], inplace=True)

# sentence_data_df_enhanced_all.reset_index(inplace=True)
# print(len(sentence_data_df_enhanced_all))

# %%
all_job_ids_with_skill_extracted_enhanced = set(sentence_data_df_enhanced_all['job id'].unique())
len(all_job_ids_with_skill_extracted_enhanced)


# %%
job_dates_skill_extracted_enhanced = {k:v for k,v in sample_job_dates.items() if k in all_job_ids_with_skill_extracted_enhanced}

num_dates_with_skill_extracted_enhanced= find_num_dates(count_dates(job_dates_skill_extracted_enhanced))


# %%
print(len(sample_job_dates))
print(len(set(sample_job_dates.keys())))

# %%
plt.figure(figsize=(10, 4))
no_none=True

plot_prop_data(num_dates, no_none, plt, label="All data", color="blue")
# plot_prop_data(num_dates_notexpired, no_none, plt, label="All data - not expired", color="black")
plot_prop_data(num_dates_sample, no_none, plt, label="Sample of data",  color="red")
# plot_prop_data(num_dates_sample_not_expired, no_none, plt, label="Sample of data with not expired files", color="yellow")
plot_prop_data(num_dates_with_skill_sents, no_none, plt, label="Sample of data with skill sentences", color="green")
# plot_prop_data(num_dates_with_embs, no_none, plt, label="Sample of data with embs", color="magenta")
# plot_prop_data(num_dates_with_red_embs, no_none, plt, label="Sample of data with reduced embs", color="yellow")
# plot_prop_data(num_dates_with_skill_extracted_enhanced, no_none, plt, label="Sample of data with skill sents - enhanced", color="purple")
# plot_prop_data(num_dates_with_skill_extracted, no_none, plt, label="Sample of data with skill sents", color="green")

plt.legend()
plt.xlabel("Date of job advert (2014 = no date given)")
plt.ylabel("Proportion")


# %%
def get_counts_by_year(counts_by_year_month):
    counts_by_year = defaultdict(int)
    for date, count in counts_by_year_month.items():
        counts_by_year[int(str(date)[0:4])] += count
    return counts_by_year


# %%
num_dates_year = get_counts_by_year(num_dates)
num_dates_notexpired_year = get_counts_by_year(num_dates_notexpired)
num_dates_sample_year = get_counts_by_year(num_dates_sample)
num_dates_sample_not_expired_year = get_counts_by_year(num_dates_sample_not_expired)
num_dates_with_skill_sents_year = get_counts_by_year(num_dates_with_skill_sents)
num_dates_with_embs_year = get_counts_by_year(num_dates_with_embs)
num_dates_with_red_embs_year = get_counts_by_year(num_dates_with_red_embs)
num_dates_with_skill_extracted_enhanced_year = get_counts_by_year(num_dates_with_skill_extracted_enhanced)


# %%
def plot_prop_data_side(dates, no_none, plt, label, color, alpha=0.5,width=0.3):
    if no_none:
        dates = {k:v for k,v in dates.items() if k!=2014}
    plt.bar(
        [d+width for d in dates.keys()],
        [v / sum(dates.values()) for v in dates.values()],
        width=0.2,
        alpha=alpha,
        color=color,
        label=label,
    )


# %%
plt.figure(figsize=(10, 4))
no_none=True

plot_prop_data_side(num_dates_year, no_none, plt, label="All data", color="blue",width=0)
plot_prop_data_side(num_dates_sample_year, no_none, plt, label="Original sample of data",  color="red",width=0.2)
plot_prop_data_side(num_dates_with_skill_sents_year, no_none, plt, label="Sample of data with sentences processed",  color="yellow",width=0.4)
plot_prop_data_side(num_dates_with_skill_extracted_enhanced_year, no_none, plt, label="Sample of data with skills extracted", color="green",width=0.6)

plt.legend(loc="lower left")
plt.xlabel("Date of job advert (2014 = no date given)")
plt.ylabel("Proportion")


# %% [markdown]
# ## 2021/21 issue
# Is it because there is simply no full text fields in that data?
# In `get_no_texts_tk_data.py` all the job ids for adverts with no text were found

# %%
all_tk_no_full_text = load_s3_data(
        s3, BUCKET_NAME, 'outputs/tk_data_analysis_new_method/all_tk_no_full_text.json'
    )

# %%
len(all_tk_no_full_text)

# %%
set_all_tk_no_full_text = set(all_tk_no_full_text)

sample_job_dates_with_full_text = {}
for job_id in tqdm(sample_job_dates):
    if job_id not in set_all_tk_no_full_text:
        sample_job_dates_with_full_text[job_id] = sample_job_dates[job_id]
        

# %%
sample_job_dates_with_full_text_count = defaultdict(int)
for k, v in tqdm(sample_job_dates_with_full_text.items()):
    if v:
        date = v[0:7]
        sample_job_dates_with_full_text_count[date] += 1
    else:
        sample_job_dates_with_full_text_count["Not given"] += 1
num_dates_sample_with_full_text = find_num_dates(sample_job_dates_with_full_text_count)

# %%
# All TK data no expired and full text
tk_dates_count_not_expired_got_full_text = load_s3_data(
    s3, BUCKET_NAME,
    "outputs/tk_data_analysis_new_method/metadata_date/tk_dates_count_not_expired_got_full_text.json"
)

# %%
all_tk_fulltext_notexpired_counts_ym = defaultdict(int)
for k, v in tqdm(tk_dates_count_not_expired_got_full_text.items()):
    if k!="Not given":
        date = k[0:7]
    else:
        date = k
    all_tk_fulltext_notexpired_counts_ym[date] += v

# %%
# All TK data no expired and not full text (despite what the name of the file says)
tk_dates_count_not_expired_no_full_text = load_s3_data(
    s3, BUCKET_NAME,
    "outputs/tk_data_analysis_new_method/metadata_date/tk_dates_count_not_expired_full_text.json"
)

# %%
tk_dates_count_not_expired_no_full_text_ym = defaultdict(int)
for k, v in tqdm(tk_dates_count_not_expired_no_full_text.items()):
    if k!="Not given":
        date = k[0:7]
    else:
        date = k
    tk_dates_count_not_expired_no_full_text_ym[date] += v

# %%
num_dates_all_not_expired_with_full_text = find_num_dates(all_tk_fulltext_notexpired_counts_ym)


# %%
num_dates_tk_dates_count_not_expired_no_full_text_ym = find_num_dates(tk_dates_count_not_expired_no_full_text_ym)


# %%
plt.figure(figsize=(10, 4))
no_none=True

plot_prop_data(num_dates, no_none, plt, label="All data", color="blue")
plot_prop_data(num_dates_all_not_expired_with_full_text, no_none, plt, label="All data all full text no expired",  color="red")
# plot_prop_data(num_dates_tk_dates_count_not_expired_no_full_text_ym, no_none, plt, label="All data all full text no expired",  color="green")

# plot_prop_data(num_dates_sample, no_none, plt, label="Sample of data",  color="red")

plot_prop_data(num_dates_sample_with_full_text, no_none, plt, label="Sample of data with full text", color="yellow")
# plot_prop_data(num_dates_with_skill_sents, no_none, plt, label="Sample of data with skill sentences", color="green")

plt.legend()
plt.xlabel("Date of job advert (2014 = no date given)")
plt.ylabel("Proportion")


# %% [markdown]
# ## Find out problem area
# Output of sentence classifier

# %%
skill_sentences_dir = "outputs/sentence_classifier/data/skill_sentences/2021.10.27/textkernel-files/"

data_paths = get_s3_data_paths(
        s3, BUCKET_NAME, skill_sentences_dir, file_types=["*.json"]
    )

unique_job_ids = set()
for data_path in tqdm(data_paths):
        data = load_s3_data(s3, BUCKET_NAME, data_path)
        for job_id in data.keys():
                unique_job_ids.add(job_id)
                
len(unique_job_ids)

# %%
sentence_classifier_job_dates = {k:v for k,v in sample_job_dates.items() if k in unique_job_ids}

# %%
len(sentence_classifier_job_dates)

# %%
sentence_classifier_job_dates_count = defaultdict(int)
for k, v in tqdm(sentence_classifier_job_dates.items()):
    if v:
        date = v[0:7]
        sentence_classifier_job_dates_count[date] += 1
    else:
        sentence_classifier_job_dates_count["No date given"] += 1

# %%
num_dates_sentence_classifier = find_num_dates(sentence_classifier_job_dates_count)

# %%
plt.figure(figsize=(10, 4))
plt.bar(
    num_dates.keys(),
    [v / sum(num_dates.values()) for v in num_dates.values()],
    width=0.1,
    alpha=0.5,
    label="All data",
)
plt.bar(
    num_dates_sample.keys(),
    [v / sum(num_dates_sample.values()) for v in num_dates_sample.values()],
    width=0.1,
    color="red",
    alpha=0.5,
    label="Sample of data",
)
plt.bar(
    num_dates_sentence_classifier.keys(),
    [v / sum(num_dates_sentence_classifier.values()) for v in num_dates_sentence_classifier.values()],
    width=0.1,
    color="green",
    alpha=0.5,
    label="Sample of data that sentence are classified from",
)
plt.legend()
plt.xlabel("Date of job advert (2014 = no date given)")
plt.ylabel("Proportion")

# %% [markdown]
# ## Some examples of job adverts from 2020:

# %%
after2020jobs = [i for i,v in sample_job_dates.items() if v and v[0:3]=='202']

# %%
a_job_adv_file = load_s3_data(s3, BUCKET_NAME, 'inputs/data/textkernel-files/semiannual/2021/2021-04-01/jobs_new.39.jsonl.gz')


# %%
a_job_adv_file[0]

# %%
lengths_2020 = [len(f.get('full_text','')) for f in a_job_adv_file]

# %%
set([f.get('date','')[0:4] for f in a_job_adv_file])

# %%
plt.hist(lengths_2020, bins=100);

# %%
a_2017_job_adv_file = load_s3_data(s3, BUCKET_NAME, 'inputs/data/textkernel-files/historical/2019/2019-11-14/jobs.0.jsonl.gz')


# %%
set([f.get('date','')[0:4] for f in a_2017_job_adv_file])

# %%
lengths_not_2020 = [len(f.get('full_text','')) for f in a_2017_job_adv_file]

# %%
plt.hist(lengths_not_2020, bins=100);

# %% [markdown]
# ## If there are duplicate job ids
#
# The data from the f"outputs/tk_data_analysis/metadata_job/{file_name}.json" locations may be warped.
#
# Check a file from the original sample and see if the dates match.

# %%
original_sample = load_s3_data(s3, BUCKET_NAME, "outputs/tk_sample_data/sample_file_locations.json")

# %%
all_sample_job_ids = [job_id for job_ids in original_sample.values() for job_id in job_ids]

# %%
# The sample has all-unique job ids
print(len(all_sample_job_ids))
print(len(set(all_sample_job_ids)))

# %% [markdown]
# #### File 0: 0.9946044996436934 dates are equal
#
# The not equal are:
#
# [(None, '2019-08-11'),
#  (None, '2019-07-25'),
#  (None, '2019-07-25'),
#  (None, '2019-03-06'),
#  (None, '2019-07-25'),
#  (None, '2019-07-25'),
#  (None, '2019-07-25'),
#  (None, '2019-07-25'),
#
# #### File 1: 0.9883720930232558 dates are equal
#
# [(None, '2018-08-22'),
#  (None, '2019-08-25'),
#  (None, '2019-06-14'),
#  (None, '2019-06-14'),
#  (None, '2019-08-25'),
#  (None, '2019-08-25'),

# %%
# [k for k in original_sample.keys() if '2021' in k]

# %%
file_name = 'semiannual/2021/2021-04-01/jobs_new.16.jsonl.gz'

job_ids = original_sample[file_name]
len(job_ids)

# %%
# file_name = list(original_sample.keys())[1]
# job_ids = original_sample[file_name]

# %%
method_dates = {job_id: sample_job_dates[job_id] for job_id in job_ids}

# %%
tk_orig_data_file = load_s3_data(s3, BUCKET_NAME, f'inputs/data/textkernel-files/{file_name}')

# %%
orig_dates = {l['job_id']: l['date'] for l in tk_orig_data_file if l['job_id'] in job_ids}

# %%
are_equal = [date==orig_dates[job_id] for job_id, date in method_dates.items()]
print(sum(are_equal)/len(method_dates))
print(len(method_dates))

# %%
[(date, orig_dates[job_id]) for job_id, date in method_dates.items() if date!=orig_dates[job_id]]

# %% [markdown]
# ## New method of saving sample dates

# %%
new_sample_job_dates = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis/metadata_date/sample_filtered_2021.11.05_new_method.json")


# %%
len(new_sample_job_dates)

# %%
len(set([s for (s,v) in new_sample_job_dates]))

# %%
len(sample_job_dates)

# %%
list(new_sample_job_dates.keys())[0]

# %%
equal_samples = [v==sample_job_dates[k] for k,v in new_sample_job_dates]

# %%
sum(equal_samples)/len(equal_samples)

# %%
Counter([(sample_job_dates[k][0:4] if sample_job_dates[k] else None) for k,v in new_sample_job_dates if v!=sample_job_dates[k] ])


# %%
Counter([(v[0:4] if v else None) for k,v in new_sample_job_dates if v!=sample_job_dates[k] ])


# %%
Counter([(k[0:4] if k else None) for k in sample_job_dates.values()])


# %% [markdown]
# ## Input of reduce embeddings

# %%
sentence_embeddings_dir = 'outputs/skills_extraction/word_embeddings/data/2021.11.05'
sentence_embeddings_dirs = get_s3_data_paths(s3, BUCKET_NAME, sentence_embeddings_dir, file_types=["*.json"])


# %%

# %%
sentence_embeddings_dirs = get_s3_data_paths(s3, BUCKET_NAME, sentence_embeddings_dir, file_types=["*.json"])

input_red_embs_job_ids = []
for embedding_dir in tqdm(sentence_embeddings_dirs):
    if "embeddings.json" in embedding_dir:
        sentence_embeddings = load_s3_data(s3, BUCKET_NAME, embedding_dir)
        for job_id, sent_id, _, _ in sentence_embeddings:
            input_red_embs_job_ids.append((job_id,sent_id))

# %%
len(input_red_embs_job_ids)

# %%
len(set(input_red_embs_job_ids))

# %%

# %%

# %%
## MAYBE BAD BELOW

# %% [markdown]
# ## Load all TK counts
# Can't remember how I did this!

# %%
all_tk_year_month_counts = pd.read_csv(
    "outputs/tk_analysis/all_tk_year_month_counts.csv"
)
all_tk_count_region_df = pd.read_csv("outputs/tk_analysis/all_tk_regions_counts.csv")
all_tk_count_subregion_df = pd.read_csv(
    "outputs/tk_analysis/all_tk_subregions_counts.csv"
)

# %%
all_tk_year_month_counts

# %% [markdown]
# ## Load the job advert info for the original 5 million sample of job adverts
# Found from `filter_bulk_data.py`

# %%
job_dates = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis/metadata_date/sample_filtered_2021.11.05.json")
job_titles = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis/metadata_job/sample_filtered_2021.11.05.json")
job_location = load_s3_data(s3, BUCKET_NAME, "outputs/tk_data_analysis/metadata_location/sample_filtered_2021.11.05.json")

# %%
print(len(job_dates))
print(len(job_titles))
print(len(job_location))

# %%
Counter([d[0:4] for d in job_dates.values() if d])

# %% [markdown]
# ## Job adverts where skills were extracted

# %%
sentence_data = load_s3_data(s3, BUCKET_NAME, "outputs/skills_extraction/extracted_skills/2021.11.05_sentences_skills_data.json")
    

# %%
sentence_data_df = pd.DataFrame(sentence_data)
sentence_data_df.head(2)

# %%
job_ids_embeddings = set(sentence_data_df['job id'].unique())
print(f"There were {len(job_ids_embeddings)} job ids which had embeddings found")
job_ids_skills = set(sentence_data_df[sentence_data_df['Cluster number predicted']>=0]['job id'].unique())
print(f"There were {len(job_ids_skills)} job ids which had skills found")

# %%
job_dates_skills = {k:v for k, v in job_dates.items() if k in job_ids_skills}
print(len(job_dates_skills))
print(set([d[0:4] for d in job_dates_skills.values() if d]))
print(Counter([d[0:4] for d in job_dates_skills.values() if d]))

# %%
job_dates_embs = {k:v for k, v in job_dates.items() if k in job_ids_embeddings}
print(len(job_dates_embs))
print(set([d[0:4] for d in job_dates_embs.values() if d]))
print(Counter([d[0:4] for d in job_dates_embs.values() if d]))

# %% [markdown]
# ## How many job adverts

# %%
total_number_jobadvs = 62892486  # Found in 'TextKernel Data.ipynb'

print(f"Sentences from original sample are from {len(job_dates)} job adverts")
print(
    f"This is {round(len(job_dates)*100/total_number_jobadvs,2)}% of all job adverts"
)

print(f"Sentences that had embeddings are from {len(job_dates_embs)} job adverts")
print(
    f"This is {round(len(job_dates_embs)*100/total_number_jobadvs,2)}% of all job adverts"
)

skill_num_jobadvs = len(job_dates_skills)
print(f"Sentences that make up skills are from {skill_num_jobadvs} job adverts")
print(
    f"This is {round(skill_num_jobadvs*100/total_number_jobadvs,2)}% of all job adverts"
)

# %% [markdown]
# ## Dates
# 'date', 'expiration_date'

# %%
job_dates['a331fec824394137a11fdb82529f998e']

# %%
## All tk data
all_tk_dates = pd.DataFrame.from_dict(job_dates, orient='index')
all_count_no_date = len(all_tk_dates[~pd.notnull(all_tk_dates[0])])
print(len(all_tk_dates))
all_tk_dates = all_tk_dates[pd.notnull(all_tk_dates[0])]
print(len(all_tk_dates))
all_tk_dates["year"] = pd.DatetimeIndex(all_tk_dates[0]).year
all_tk_dates["month"] = pd.DatetimeIndex(all_tk_dates[0]).month

# Group by year+month
all_year_month_counts = all_tk_dates.groupby(["year", "month"])[0].count()
all_year_month_counts = all_year_month_counts.sort_index().reset_index()
all_year_month_counts["year/month"] = (
    all_year_month_counts[["year", "month"]].astype(str).agg("/".join, axis=1)
)
# Add back in the none counts
all_year_month_counts = pd.concat(
    [all_year_month_counts, pd.DataFrame({'year/month':['Not given'], 'year':['Not given'], 0:[all_count_no_date]})])

all_year_month_counts["Proportion"] = all_year_month_counts[0] / (all_year_month_counts[0].sum())


# %%
## Embs tk data
embs_tk_dates = pd.DataFrame.from_dict(job_dates_embs, orient='index')
count_no_date_embs = len(embs_tk_dates[~pd.notnull(embs_tk_dates[0])])
print(len(embs_tk_dates))
embs_tk_dates = embs_tk_dates[pd.notnull(embs_tk_dates[0])]
print(len(embs_tk_dates))
embs_tk_dates["year"] = pd.DatetimeIndex(embs_tk_dates[0]).year
embs_tk_dates["month"] = pd.DatetimeIndex(embs_tk_dates[0]).month

# Group by year+month
embs_year_month_counts = embs_tk_dates.groupby(["year", "month"])[0].count()
embs_year_month_counts = embs_year_month_counts.sort_index().reset_index()
embs_year_month_counts["year/month"] = (
    embs_year_month_counts[["year", "month"]].astype(str).agg("/".join, axis=1)
)
# Add back in the none counts
embs_year_month_counts = pd.concat(
    [embs_year_month_counts, pd.DataFrame({'year/month':['Not given'], 'year':['Not given'], 0:[count_no_date_embs]})])

embs_year_month_counts["Proportion"] = embs_year_month_counts[0] / (embs_year_month_counts[0].sum())


# %%
## Embs tk data
skills_tk_dates = pd.DataFrame.from_dict(job_dates_skills, orient='index')
count_no_date_skills = len(skills_tk_dates[~pd.notnull(skills_tk_dates[0])])
print(len(skills_tk_dates))
skills_tk_dates = skills_tk_dates[pd.notnull(skills_tk_dates[0])]
print(len(skills_tk_dates))
skills_tk_dates["year"] = pd.DatetimeIndex(skills_tk_dates[0]).year
skills_tk_dates["month"] = pd.DatetimeIndex(skills_tk_dates[0]).month

# Group by year+month
skills_year_month_counts = skills_tk_dates.groupby(["year", "month"])[0].count()
skills_year_month_counts = skills_year_month_counts.sort_index().reset_index()
skills_year_month_counts["year/month"] = (
    skills_year_month_counts[["year", "month"]].astype(str).agg("/".join, axis=1)
)
# Add back in the none counts
skills_year_month_counts = pd.concat(
    [skills_year_month_counts, pd.DataFrame({'year/month':['Not given'], 'year':['Not given'], 0:[count_no_date_skills]})])

skills_year_month_counts["Proportion"] = skills_year_month_counts[0] / (skills_year_month_counts[0].sum())


# %% [markdown]
# ### Get proportions for side by side comparison
# Not using no-date data

# %%
all_tk_year_month_counts.fillna('Not given', inplace=True)

# %%
all_tk_year_month_counts.columns

# %%
all_tk_year_month_counts["Proportion"] = all_tk_year_month_counts["0"] / (
    all_tk_year_month_counts["0"].sum()
)


# %% [markdown]
# ### Plot dates with all TK dates

# %%
nesta_orange = [255 / 255, 90 / 255, 0]
nesta_purple = [155 / 255, 0, 195 / 255]
nesta_grey = [165 / 255, 148 / 255, 130 / 255]

# %%
all_tk_year_month_counts.head(2)

# %%
# Set the year/month missing counts as 0

missing_years = list(set(all_tk_year_month_counts['year/month'].tolist()).difference(set(skills_year_month_counts['year/month'].tolist())))
skills_year_month_counts_1 = pd.concat(
    [skills_year_month_counts, pd.DataFrame({'year/month':missing_years, 0:[0]*len(missing_years)})])

missing_years = list(set(all_tk_year_month_counts['year/month'].tolist()).difference(set(embs_year_month_counts['year/month'].tolist())))
embs_year_month_counts_1 = pd.concat(
    [embs_year_month_counts, pd.DataFrame({'year/month':missing_years, 0:[0]*len(missing_years)})])


ax = all_tk_year_month_counts.plot(
    x="year/month",
    y="Proportion",
    xlabel="Date of job advert",
    ylabel="Proportion of job adverts",
    c=nesta_grey,
    label="All TK job adverts"
)

ax = all_year_month_counts.plot(
    x="year/month",
    y="Proportion",
    xlabel="Date of job advert",
    ylabel="Proportion of job adverts",
    c="purple",
    ax=ax,
    label="Original sample of job adverts"
)
ax = embs_year_month_counts_1.plot(
    x="year/month",
    y="Proportion",
    c='red',
    ax=ax,
    label="Job adverts in embeddings (inc -2)"
)
ax = skills_year_month_counts_1.plot(
    x="year/month",
    y="Proportion",
    c="orange",
    ax=ax,
    label="Job adverts in skills"
)

# ax.legend(["All TK job adverts", "TK job adverts in sample", "c",'d'])
ax.legend()
ax.figure.savefig(
    f"outputs/tk_analysis/{file_date}_job_ad_date_sample_comparison.pdf", bbox_inches="tight"
)

# %%
# all_tk_year_month_counts_nonull["year"] = all_tk_year_month_counts_nonull[
#     "year"
# ].astype(str)
# all_year_month_counts["year"] = all_year_month_counts[
#     "year"
# ].astype(str)
# skills_year_month_counts["year"] = skills_year_month_counts[
#     "year"
# ].astype(str)
# embs_year_month_counts["year"] = embs_year_month_counts[
#     "year"
# ].astype(str)

# %%
# all_tk_year_month_counts_nonull
# all_year_month_counts
# skills_year_month_counts
# embs_year_month_counts

# %%
all_year_month_counts

# %%
fig = plt.figure(figsize=(7, 4))  # Create matplotlib figure

ax = fig.add_subplot(111)  # Create matplotlib axes

width = 0.2

# Set the year missing counts as 0

missing_years = list(set(all_tk_year_month_counts['year'].tolist()).difference(set(skills_year_month_counts['year'].tolist())))
skills_year_month_counts_2 = pd.concat(
    [skills_year_month_counts, pd.DataFrame({'year':missing_years, 0:[0]*len(missing_years)})])

missing_years = list(set(all_tk_year_month_counts['year'].tolist()).difference(set(embs_year_month_counts['year'].tolist())))
embs_year_month_counts_2 = pd.concat(
    [embs_year_month_counts, pd.DataFrame({'year':missing_years, 0:[0]*len(missing_years)})])

ax = pd.DataFrame(
    all_tk_year_month_counts.groupby("year")["0"].sum()
    / sum(all_tk_year_month_counts["0"])
).plot.bar(color=nesta_grey, ax=ax, width=width, position=1)

ax=pd.DataFrame(
    all_year_month_counts.groupby("year")[0].sum() / sum(all_year_month_counts[0])
).plot.bar(color="purple", ax=ax, width=width, position=2)

ax=pd.DataFrame(
    embs_year_month_counts_2.groupby("year")[0].sum() / sum(embs_year_month_counts[0])
).plot.bar(color="red", ax=ax, width=width, position=3)


ax=pd.DataFrame(
    skills_year_month_counts_2.groupby("year")[0].sum() / sum(skills_year_month_counts[0])
).plot.bar(color="orange", ax=ax, width=width, position=4)


ax.set_ylabel("Proportion of job adverts")
ax.set_xlabel("Year of job advert")
ax.legend([
    "All TK job adverts", "Original sample of job adverts", "Job adverts in embeddings (inc -2)","Job adverts in skills"
], loc="upper right")

ax.figure.savefig(
    f"outputs/tk_analysis/{file_date}_job_ad_year_sample_comparison.pdf", bbox_inches="tight"
)

# %% [markdown]
# # UP TO HERE

# %% [markdown]
# ## Location

# %%
tk_region = []
tk_subregion = []
for file_name in tqdm(range(0, 13)):
    file_dict = load_s3_data(
        s3, bucket_name, f"outputs/tk_data_analysis/metadata_location/{file_name}.json"
    )
    tk_region.extend(
        [f[2] for job_id, f in file_dict.items() if f and job_id in skill_job_ads]
    )
    tk_subregion.extend(
        [f[3] for job_id, f in file_dict.items() if f and job_id in skill_job_ads]
    )

print(len(tk_region))
print(len(tk_subregion))

# %%
print(len(set(tk_region)))
print(len(set(tk_subregion)))

# %%
count_region_df = pd.DataFrame.from_dict(Counter(tk_region), orient="index")
count_region_df

# %%
count_region_df.to_csv("outputs/tk_analysis/skills_tk_regions_counts.csv")

# %%
print(count_region_df[0].sum())
count_region_df = count_region_df[pd.notnull(count_region_df.index)]
print(count_region_df[0].sum())

# %%
count_region_df

# %%
all_tk_count_region_df_nonull = all_tk_count_region_df[
    pd.notnull(all_tk_count_region_df["Unnamed: 0"])
]
all_tk_count_region_df_nonull.index = all_tk_count_region_df_nonull["Unnamed: 0"]
all_tk_count_region_df_nonull

# %%
fig = plt.figure(figsize=(7, 4))  # Create matplotlib figure

ax = fig.add_subplot(111)  # Create matplotlib axes

width = 0.3

ax = (
    pd.DataFrame(
        all_tk_count_region_df_nonull["0"] / sum(all_tk_count_region_df_nonull["0"])
    )
    .sort_values(by=["0"], ascending=False)
    .plot.bar(color=nesta_grey, legend=False, ax=ax, width=width, position=1)
)

ax = pd.DataFrame(count_region_df[0] / sum(count_region_df[0])).plot.bar(
    color=nesta_orange, legend=False, ax=ax, width=width, position=0
)

ax.set_ylabel("Proportion of job adverts")
ax.set_xlabel("Region of job advert")
ax.legend(["All TK job adverts", "TK job adverts in sample"], loc="upper right")

ax.figure.savefig(
    "outputs/tk_analysis/job_ad_region_sample_comparison.pdf", bbox_inches="tight"
)

# %%
count_subregion_df = pd.DataFrame.from_dict(Counter(tk_subregion), orient="index")

# %%
count_subregion_df.to_csv("outputs/tk_analysis/skills_tk_subregions_counts.csv")

# %%
print(count_subregion_df[0].sum())
count_subregion_df = count_subregion_df[pd.notnull(count_subregion_df.index)]
print(count_subregion_df[0].sum())

# %%
all_tk_count_subregion_df_nonull = all_tk_count_subregion_df[
    pd.notnull(all_tk_count_subregion_df["Unnamed: 0"])
]
all_tk_count_subregion_df_nonull.index = all_tk_count_subregion_df_nonull["Unnamed: 0"]

# %%
prop_subregions_all = pd.DataFrame(
    all_tk_count_subregion_df_nonull["0"] / sum(all_tk_count_subregion_df_nonull["0"])
)
prop_subregions_sample = pd.DataFrame(
    count_subregion_df[0] / sum(count_subregion_df[0])
)

# %%
top_50_subregions_all = prop_subregions_all.sort_values(by=["0"], ascending=False)[
    0:50
].index

# %%
fig = plt.figure(figsize=(14, 4))  # Create matplotlib figure

ax = fig.add_subplot(111)  # Create matplotlib axes

width = 0.3


ax = prop_subregions_all.loc[top_50_subregions_all].plot.bar(
    color=nesta_grey, legend=False, ax=ax, width=width, position=1
)

ax = prop_subregions_sample.loc[top_50_subregions_all].plot.bar(
    color=nesta_orange, legend=False, ax=ax, width=width, position=0
)

ax.set_ylabel("Proportion of job adverts")
ax.set_xlabel("Subregion of job advert")
ax.legend(["All TK job adverts", "TK job adverts in sample"], loc="upper right")

ax.figure.savefig(
    "outputs/tk_analysis/job_ad_subregion_sample_comparison.pdf", bbox_inches="tight"
)

# %% [markdown]
# ## Plots together

# %%
width = 0.3

plt.figure(figsize=(12, 8))

ax3 = plt.subplot(212)
prop_subregions_all.loc[top_50_subregions_all].plot.bar(
    color=nesta_grey, legend=False, ax=ax3, width=width, position=1
)
prop_subregions_sample.loc[top_50_subregions_all].plot.bar(
    color=nesta_orange, legend=False, ax=ax3, width=width, position=0
)
ax3.set_ylabel("Proportion of job adverts")
ax3.set_xlabel("Subregion of job advert")

ax1 = plt.subplot(221)
pd.DataFrame(
    all_tk_year_month_counts_nonull.groupby("year")["0"].sum()
    / sum(all_tk_year_month_counts_nonull["0"])
).plot.bar(color=nesta_grey, ax=ax1, width=width, position=1, legend=False)
pd.DataFrame(
    year_month_counts.groupby("year")[0].sum() / sum(year_month_counts[0])
).plot.bar(color=nesta_orange, ax=ax1, width=width, position=0, legend=False)
ax1.set_ylabel("Proportion of job adverts")
ax1.set_xlabel("Year of job advert")

ax2 = plt.subplot(222)
pd.DataFrame(
    all_tk_count_region_df_nonull["0"] / sum(all_tk_count_region_df_nonull["0"])
).sort_values(by=["0"], ascending=False).plot.bar(
    color=nesta_grey, legend=False, ax=ax2, width=width, position=1
)
pd.DataFrame(count_region_df[0] / sum(count_region_df[0])).plot.bar(
    color=nesta_orange, legend=False, ax=ax2, width=width, position=0
)
ax2.set_ylabel("Proportion of job adverts")
ax2.set_xlabel("Region of job advert")
ax2.legend(["All TK job adverts", "TK job adverts in sample"], loc="upper right")

plt.tight_layout()
plt.savefig(
    "outputs/tk_analysis/job_ad_together_sample_comparison.pdf", bbox_inches="tight"
)

# %%
width = 0.3

plt.figure(figsize=(12, 8))

ax1 = plt.subplot(221)
pd.DataFrame(
    all_tk_year_month_counts_nonull.groupby("year")["0"].sum()
    / sum(all_tk_year_month_counts_nonull["0"])
).plot.bar(color=nesta_grey, ax=ax1, width=width, position=1, legend=False)
pd.DataFrame(
    year_month_counts.groupby("year")[0].sum() / sum(year_month_counts[0])
).plot.bar(color=nesta_orange, ax=ax1, width=width, position=0, legend=False)
ax1.set_ylabel("Proportion of job adverts")
ax1.set_xlabel("Year of job advert")

ax2 = plt.subplot(222)
pd.DataFrame(
    all_tk_count_region_df_nonull["0"] / sum(all_tk_count_region_df_nonull["0"])
).sort_values(by=["0"], ascending=False).plot.bar(
    color=nesta_grey, legend=False, ax=ax2, width=width, position=1
)
pd.DataFrame(count_region_df[0] / sum(count_region_df[0])).plot.bar(
    color=nesta_orange, legend=False, ax=ax2, width=width, position=0
)
ax2.set_ylabel("Proportion of job adverts")
ax2.set_xlabel("Region of job advert")
ax2.legend(["All TK job adverts", "TK job adverts in sample"], loc="upper right")

plt.tight_layout()
plt.savefig(
    "outputs/tk_analysis/job_ad_together_sample_comparison_two.pdf", bbox_inches="tight"
)

# %%

# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Automate training data from TextKernel data
#
# Can we get more training data for the sentence classifier by using headings?
# e.g. every sentence between two headings is a skill sentence (e.g. list of skills)
#
# Perhaps certain job urls have good formatting of the skills/background section?
#
# Use textkernel data

import json
from collections import Counter

with open('../../../../inputs/TextKernel_sample/jobs_new.1.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

data[0].keys()

# ## Conditions text - no skills text
# I think everything in 'conditions_description' is not a skill? They don't all have this field though.

print(len([d for d in data if d.get('conditions_description')]))
print(len([d for d in data if not d.get('conditions_description')]))

conditions_texts = [d['conditions_description']
                    for d in data if d.get('conditions_description')]

conditions_texts[100]

with open('../../../../outputs/sentence_classifier/data/generated_training_data/TextKernel_no_skills_sections.json', 'w') as file:
    json.dump(conditions_texts, file)

# ## Candidate description - Skills in text
#
# 'candidate_description' has the skills, but not all about skills (e.g. 'You will have a minimum of 2 years' experience'). So see if there any sub sections to it - ways of splitting it up into just skills chunks of text.

print(len([d for d in data if d.get('candidate_description')]))
print(len([d for d in data if not d.get('candidate_description')]))

candidate_description_texts = [d['candidate_description']
                               for d in data if d.get('candidate_description')]

candidate_description_texts[0]


# ### Try to find common combinations of words (to find possible headings)

def token_sliding_window(text, sizes=[2, 3, 4]):
    tokens = text.split(' ')
    for size in sizes:
        for i in range(len(tokens) - size + 1):
            if len(tokens[i]) > 0:
                if tokens[i][0].isupper():
                    # First char has to be upper case
                    yield ' '.join(tokens[i: i + size])


merged_texts = '. '.join([data[i]['full_text'] for i in range(0, 100000)])
possible_headings = [t for t in token_sliding_window(merged_texts, [2, 3, 4])]

counter_possible_headings = Counter(possible_headings)

counter_possible_headings.most_common(2)

# 'headings' with the word 'skill' in
possible_skill_headings = {
    c: v for c, v in counter_possible_headings.items() if 'skill' in c.lower()
}
possible_skill_headings = {
    k: v for k, v in sorted(
        possible_skill_headings.items(), key=lambda item: item[1], reverse=True
    )
}

list(possible_skill_headings.items())[:20]

counter_possible_headings.most_common(10)

# I think the 'Required skills' heading might be a good place to start
skill_jd = [d for d in data if 'Required skills' in d['full_text']]
len(skill_jd)

Counter([s['source_website'] for s in skill_jd]).most_common(10)

skill_jd_reed = [d for d in skill_jd if d['source_website'] == 'reed.co.uk']

# ### Assume jobs from Reed with the text 'Required skills' has skills listed in a section between ''Required skills' and the first \n\n
# Extract that whole bit and save out.

tk_skills_data = []
for jd in skill_jd_reed:
    text = jd['full_text']
    skills_text = text.split('Required skills')[1].split('\n\n')[1]
    tk_skills_data.append(skills_text)

len(tk_skills_data)

with open('../../../../outputs/sentence_classifier/data/generated_training_data/TextKernel_skills_texts.json', 'w') as file:
    json.dump(tk_skills_data, file)

#!/bin/bash
set -eo pipefail

source "$PWD/.env.shared"
source "$PWD/bin/conda_activate.sh"

# Fetch description of package (setup.py and conda make this horrible)
DESCRIPTION=$(conda_activate && pip show $PROJECT_NAME | grep Summary | cut -d " " -f2-)

# Create
gh repo create $GITHUB_ACCOUNT/$PROJECT_NAME --$PROJECT_OPENNESS -d "$DESCRIPTION" -y

# push initial branches
git push --all

# Set:
# - default branch to `dev`
# - squash merge as only option
gh api -X PATCH\
 -F default_branch=dev\
 -F allow_merge_commit=false\
 -F allow_rebase_merge=false\
  repos/$GITHUB_ACCOUNT/$PROJECT_NAME --silent

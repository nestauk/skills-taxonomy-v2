#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(pwd)

# Fetch research daps key
aws s3 cp s3://nesta-production-config/research_daps.key . --quiet

# Clone research daps
cd /tmp
\rm -rf /tmp/research_daps
git clone git@github.com:nestauk/research_daps.git -q
cd research_daps

# Unencrypt research daps
git-crypt unlock $PROJECT_DIR/research_daps.key &> /dev/null

# Copy metaflow config, TODO: backup an existing config
mkdir -p ~/.metaflowconfig
cp research_daps/config/metaflowconfig/config.json ~/.metaflowconfig/config_ds-cookiecutter.json

# Clean up
\rm -rf /tmp/research_daps
cd $PROJECT_DIR
rm research_daps.key

#!/bin/bash

function get_metaflow_config() {
    # Clean up on success or failure
    trap "{ rm -rf /tmp/research_daps /tmp/research_daps.key; }" \
    SIGINT SIGTERM ERR EXIT

    # Fetch research daps key
    # Clone research daps
    # Unencrypt research daps
    # Copy metaflow config
    aws s3 cp s3://nesta-production-config/research_daps.key /tmp/research_daps.key \
    && git clone git@github.com:nestauk/research_daps.git /tmp/research_daps \
    && pushd /tmp/research_daps \
    && git-crypt unlock "../research_daps.key" \
    && mkdir -p "$HOME/.metaflowconfig" \
    && cp research_daps/config/metaflowconfig/config.json "$HOME/.metaflowconfig/config_ds-cookiecutter.json" \
    && popd || return
}
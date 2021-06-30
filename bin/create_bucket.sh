#!/bin/bash
set -euo pipefail

source "$PWD/.env.shared"

function create_bucket() {
    echo "Attempting creation of bucket $BUCKET ...";
    aws s3api create-bucket\
     --bucket $BUCKET\
     --region eu-west-2\
     --create-bucket-configuration\
     'LocationConstraint=eu-west-2';
}

function make_bucket_private() {
    echo "Configuring $BUCKET to block all public access...";
    aws s3api put-public-access-block\
     --bucket $BUCKET\
     --public-access-block-configuration\
     "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true";
}

create_bucket && make_bucket_private || echo "ERROR: Bucket creation failed!"

function conda_activate() {
    source .env.shared \
     && source $(conda info --base)/etc/profile.d/conda.sh  \
     && conda activate $PROJECT_NAME \
     || source activate $PROJECT_NAME;
}

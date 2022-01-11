import time
import logging
from multiprocessing import Pool
from functools import partial
from collections import defaultdict

from skills_taxonomy_v2.pipeline.sentence_classifier.utils import split_sentence, make_chunks, split_sentence_over_chunk
from skills_taxonomy_v2 import BUCKET_NAME
from skills_taxonomy_v2.getters.s3_data import load_s3_data
from skills_taxonomy_v2.pipeline.sentence_classifier.predict_sentence_class import predict_sentences

logger = logging.getLogger(__name__)

def run_predict(
        s3, data_path, job_ids_set, sent_classifier
    ):
    """
    Get predictions from the sample of job adverts already found.
    Don't re-run any of the job ids in done_job_ids.
    """
  
    # Run predictions and save outputs iteratively
    logger.info(f"Loading data from {data_path} ...")
    data = load_s3_data(s3, BUCKET_NAME, data_path)

    # Sample of job adverts used for this file
    neccessary_fields = ["full_text", "job_id"]
    data = [
        {field: job_ad.get(field, None) for field in neccessary_fields}
        for job_ad in data
        if job_ad["job_id"] in job_ids_set
    ]

    if data:
        logger.info(f"Splitting sentences ...")
        start_time = time.time()
        with Pool(4) as pool:  # 4 cpus
            chunks = make_chunks(data, 1000)  # chunks of 1000s sentences
            partial_split_sentence = partial(split_sentence_over_chunk, min_length=30)
            # NB the output will be a list of lists, so make sure to flatten after this!
            split_sentence_pool_output = pool.map(partial_split_sentence, chunks)
        logger.info(f"Splitting sentences took {time.time() - start_time} seconds")

        # Flatten and process output into one list of sentences for all documents
        start_time = time.time()
        sentences = []
        job_ids = []
        for chunk_split_sentence_pool_output in split_sentence_pool_output:
            for job_id, s in chunk_split_sentence_pool_output:
                if s:
                    sentences += s
                    job_ids += [job_id] * len(s)
        logger.info(f"Processing sentences took {time.time() - start_time} seconds")

        if sentences:
            logger.info(f"Transforming skill sentences ...")
            sentences_vec = sent_classifier.transform(sentences)
            pool_sentences_vec = [
                (vec_ix, [vec]) for vec_ix, vec in enumerate(sentences_vec)
            ]

            logger.info(f"Chunking up sentences ...")
            start_time = time.time()
            # Manually chunk up the data to predict multiple in a pool
            # This is because predict can't deal with massive vectors
            pool_sentences_vecs = []
            pool_sentences_vec = []
            for vec_ix, vec in enumerate(sentences_vec):
                pool_sentences_vec.append((vec_ix, vec))
                if len(pool_sentences_vec) > 1000:
                    pool_sentences_vecs.append(pool_sentences_vec)
                    pool_sentences_vec = []
            if len(pool_sentences_vec) != 0:
                # Add the final chunk if not empty
                pool_sentences_vecs.append(pool_sentences_vec)
            logger.info(
                f"Chunking up sentences into {len(pool_sentences_vecs)} chunks took {time.time() - start_time} seconds"
            )

            logger.info(f"Predicting skill sentences ...")
            start_time = time.time()
            with Pool(4) as pool:  # 4 cpus
                partial_predict_sentences = partial(
                    predict_sentences, sent_classifier=sent_classifier
                )
                predict_sentences_pool_output = pool.map(
                    partial_predict_sentences, pool_sentences_vecs
                )
            logger.info(
                f"Predicting on {len(sentences)} sentences took {time.time() - start_time} seconds"
            )

            # Process output into one list of sentences for all documents
            logger.info(f"Combining data for output ...")
            start_time = time.time()
            skill_sentences_dict = defaultdict(list)
            for chunk_output in predict_sentences_pool_output:
                for (sent_ix, pred) in chunk_output:
                    if pred == 1:
                        job_id = job_ids[sent_ix]
                        sentence = sentences[sent_ix]
                        skill_sentences_dict[job_id].append(sentence)
            logger.info(f"Combining output took {time.time() - start_time} seconds")

            return skill_sentences_dict

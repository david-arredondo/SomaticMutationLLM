#!/usr/bin/env bash

BATCH_SIZE=1024  # Start with 1024, fallback in powers of 2
MIN_BATCH_SIZE=1
PKL_PATH='/data/dandreas/SomaticMutationsLLM/aa/canonical_ref.pkl'
EMB_PATH='/data/dandreas/SomaticMutationsLLM/aa/canonical_ref_randSNP_esm3.npy'

while true
do
    echo "Running randSNP ESM3 embedding script with BATCH_SIZE=${BATCH_SIZE}"
    
    python embeddings_esm3_randSNP_sorted.py \
        --pkl-path $PKL_PATH \
        --emb-path $EMB_PATH \
        --batch-size $BATCH_SIZE \
        --n-mutations 33

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Embedding script finished successfully!"
        break
    else
        echo "Script failed with exit code $EXIT_CODE, reducing batch size by 50%."
        BATCH_SIZE=$(( BATCH_SIZE / 2 ))  # reduce by half
        if [ $BATCH_SIZE -lt $MIN_BATCH_SIZE ]; then
            echo "Batch size fell below ${MIN_BATCH_SIZE}, aborting."
            exit 1
        fi
    fi
done

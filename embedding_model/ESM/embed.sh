#!/usr/bin/env bash

BATCH_SIZE=1 #power of 2 to guarantee it ends on 1
ESM_MODEL='esmC'
PY_SCRIPT='embeddings_esm3C_sorted.py'
# PY_SCRIPT='embeddings_esm3C_randSNP_sorted.py'

SEQ_PATH='/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_seqs.txt'
EMB_PATH='/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_cls_embeddings_esmC.npy'
# SEQ_PATH='/data/dandreas/SomaticMutationsLLM/aa/canonical_ref_seqs.txt'
# EMB_PATH='/data/dandreas/SomaticMutationsLLM/aa/canonical_ref_cls_embeddings_esmC.npy'
# EMB_PATH='/data/dandreas/SomaticMutationsLLM/aa/canonical_ref_randSNP_esmC.npy'

while true
do
    echo "Running embedding script with BATCH_SIZE=${BATCH_SIZE}"
    
    # Run the Python script, passing --batch-size
    python3 "$PY_SCRIPT" \
        --seqs-path $SEQ_PATH \
        --emb-path $EMB_PATH \
        --batch-size $BATCH_SIZE \
        --esm-model $ESM_MODEL 
        # --n-mutations 33 #optional, default is 33

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Embedding script finished successfully!"
        break
    else
        echo "Script failed with exit code $EXIT_CODE, reducing batch size by 50%."
        BATCH_SIZE=$(( BATCH_SIZE / 2 ))  # Reduce by 50%
        
        if [ $BATCH_SIZE -lt 1 ]; then
            echo "Batch size fell below ${MIN_BATCH_SIZE}, aborting."
            exit 1
        fi
    fi
done

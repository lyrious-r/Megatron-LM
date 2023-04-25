#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Warning: exactly one argument (input file) should be provided"
    exit 1
fi

FLAN_INPUT_PATH=$1
FILE_NAME=$(basename $FLAN_INPUT_PATH)
FILE_NAME_WITHOUT_EXT=${FILE_NAME%.*}
FLAN_DIR=$(dirname $(realpath $FLAN_INPUT_PATH))

echo INPUT: $FLAN_INPUT_PATH
echo OUTPUT DIR: $FLAN_DIR

read -p "Are you sure you want to proceed? (y/n) " CONFIRMATION
if [[ $CONFIRMATION =~ ^[Yy]$ ]]; then
    # for t5 models
    python3 tools/preprocess_data.py \
            --input $FLAN_INPUT_PATH \
            --output-prefix ${FLAN_DIR}/${FILE_NAME_WITHOUT_EXT}_t5 \
            --vocab ./vocabs/t5-base-vocab.txt \
            --dataset-impl mmap \
            --tokenizer-type BertWordPieceLowerCase \
            --workers 64 \
            --chunk-size 100 \
            --json-keys inputs targets \
            --is-supervised \
            --n-samples 1000000

    # for gpt models
    python3 tools/preprocess_data.py \
            --input $FLAN_INPUT_PATH \
            --output-prefix ${FLAN_DIR}/${FILE_NAME_WITHOUT_EXT}_gpt \
            --vocab ./vocabs/gpt2-vocab.json \
            --merge-file ./vocabs/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --workers 64 \
            --chunk-size 100 \
            --json-keys inputs targets \
            --is-supervised \
            --append-eod \
            --n-samples 1000000
else
    echo "Aborted."
fi
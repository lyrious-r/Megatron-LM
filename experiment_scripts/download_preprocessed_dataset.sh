# this script will download the preprocessed datasets
# from s3 bucket and store them in the dataset folder
BUCKET_PREFIX=https://dynapipe-ae-dataset-public.s3.us-west-2.amazonaws.com

function download_to_prefix {
    wget ${BUCKET_PREFIX}/$1 -P /root/Megatron-LM/datasets
}

download_to_prefix flan_zsopt_gpt_inputs_document.bin
download_to_prefix flan_zsopt_gpt_inputs_document.idx
download_to_prefix flan_zsopt_gpt_targets_document.bin
download_to_prefix flan_zsopt_gpt_targets_document.idx
download_to_prefix flan_zsopt_t5_inputs_document.bin
download_to_prefix flan_zsopt_t5_inputs_document.idx
download_to_prefix flan_zsopt_t5_targets_document.bin
download_to_prefix flan_zsopt_t5_targets_document.idx

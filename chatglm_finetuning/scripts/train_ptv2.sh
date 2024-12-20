#!/usr/bin/env bash

export trainer_backend=pl

train_file="../config/train_${trainer_backend}.yaml"

# enforce overwrite the config file
export train_file=${train_file}
export enable_deepspeed=false
export enable_ptv2=true
export enable_lora=false
export load_in_bit=0

# export CUDA_VISIBLE_DEVICES=0,1,2,3

usage() { echo "Usage: $0 [-m <train|dataset>]" 1>&2; exit 1; }


while getopts m: opt
do
	case "${opt}" in
		m) mode=${OPTARG};;
    *)
      usage
      ;;
	esac
done

if [ "${mode}" != "dataset" ]  && [ "${mode}" != "train" ] ; then
    usage
fi


if [[ "${mode}" == "dataset" ]] ; then
    python ../data_utils.py
    exit 0
fi

if [[ "${trainer_backend}" == "pl" ]] ; then

  python ../train.py
elif [[ "${trainer_backend}" == "cl" ]] ; then

  colossalai run --nproc_per_node 1 --num_nodes 1 ../train.py
else
  torchrun --nproc_per_node 1 --nnodes 1 ../train.py
fi
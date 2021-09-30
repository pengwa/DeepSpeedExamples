#! /bin/bash

# Change for multinode config
MP_SIZE=1
rm checkpoints/gpt2_345m_ds/* -rf
DEBUG=1
USE_ORT=1
USE_ORT_OPT=1
if [[ ${DEBUG} == 1 ]];  then
       MP_SIZE=2
       NUM_WORKERS=1
       NUM_GPUS_PER_WORKER=8
       HIDDEN_SIZE=1024
       NUM_ATTN_HEADS=16
       NUM_LAYERS=24
       BATCHSIZE=4
else
       NUM_WORKERS=${DLTS_NUM_WORKER}
       NUM_GPUS_PER_WORKER=${DLTS_NUM_GPU_PER_WORKER}
       HIDDEN_SIZE=8192
       NUM_ATTN_HEADS=32
       NUM_LAYERS=50
       BATCHSIZE=4

       #HIDDEN_SIZE=4096
       #NUM_LAYERS=24 # 50
       #BATCHSIZE=16
fi


BASE_DATA_PATH=/home/pengwa/data
DATA_PATH=${BASE_DATA_PATH}/indexed/my-gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m_ds

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
if [[ -z $1 ]]; then
       config_json="$script_dir/ds_zero_stage_1_config.json"

       # offloads to NVMe
       #config_json="$script_dir/ds_zero_stage_infinity_config.json"
else
       config_json=$script_dir/`basename $1`
fi

#ZeRO Configs
stage=0
reduce_scatter=false
contigious_gradients=false
rbs=50000000
agbs=5000000000

#Activation Checkpointing and Contigious Memory
chkp_layers=1
PA=false
PA_CPU=false
CC=false
SYNCHRONIZE=false
PROFILE=false

# TiledLinear splits, 0 is disable
TILED_LINEAR="false"
TILE_DIM=1


# Megatron Model Parallelism
LOGDIR="tboard-zero1/stage${stage}-lazyscatter-${NUM_LAYERS}l_${HIDDEN_SIZE}h_${NUM_WORKERS}n_${NUM_GPUS_PER_WORKER}g_${MP_SIZE}mp_${BATCHSIZE}b"


gpt_options=" \
        --model-parallel-size ${MP_SIZE} \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --batch-size $BATCHSIZE \
        --train-iters 100 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 2000 \
        --eval-iters 100 \
        --fp16 \
"
#	--bias-dropout-fusion \
#--bias-gelu-fusion \
#        --scattered-embeddings \
ort_options=""
log_postfix_name="pt"
if [[ ${USE_ORT} == 1 ]];  then
        ort_options="\
                --use-ort \
        "
  log_postfix_name="ort"
fi

if [[ ${USE_ORT_OPT} == 1 ]];  then
        ort_options="${ort_options}\
                --use-ort-opt \
        "
fi

#
#        --split-transformers \
        #--tensorboard-dir ${LOGDIR}
stage=1
deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

if [ "${TILED_LINEAR}" = "true" ]; then
tile_opt="${tile_opt} \
        --memory-centric-tiled-linear \
        --tile-factor=${TILE_DIM}"
fi

full_options="${gpt_options} ${deepspeed_options} ${ort_options}"

prefix="/home/pengwa/nsight-systems-2021.3.1/bin/nsys profile -o zero1_"$NUM_LAYERS"_"$MP_SIZE"_layers_%p_${log_postfix_name} -t cuda,nvtx "
run_cmd="$prefix deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}  pretrain_gpt2.py ${@:2} ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

#!/bin/bash
source ~/anaconda3/bin/activate SurfDock

export TORCH_DISTRIBUTED_DEBUG=INFO

export CUDA_VISIBLE_DEVICES="0,3,4,6"
# path of precomputed arrays
export precomputed_arrays="~/precomputed_arrays"
export CUDA_LAUNCH_BLOCKING=1
save_dir="~/surface"
score_version=3
ns=48
nv=10
num_conv_layers=6
batch_size=12
time=`date +%Y_%m_%d_%H_%M_%S`
project="V${score_version}_PDBBind_ema_model_pocket_8A"

command=`accelerate launch \
--multi_gpu \
--main_process_port 29516 \
--num_processes 4 \
~/train_accelarete.py \
--project ${project} \
--restart_dir ${save_dir}/workdir/V3_PDBBind_ema_model_pocket_8A_RunTime_2023_10_06_05_55_48_ns_48_nv_10_layer_batch_size_12_62023-10-06_05-56-09 \
--run_name project_${project}_RunTime_${time}_ns_${ns}_nv_${nv}_layer_batch_size_${batch_size}_${num_conv_layers} \
--data_dir ~/PDBBIND/PDBBind_pocket_8A/ \
--cache_path ~/PDBBIND/cache_RTMScoreFeature_Surface_PDBBIND_pocket_8A \
--surface_path ~/PDBBind_processed_8A_surface/ \
--esm_embeddings_path ~/PDBBIND/esm_embedding/esm_embedding_pocket_for_train/esm2_3billion_embeddings.pt \
--split_test ~/data/splits/timesplit_test \
--split_train ~/data/splits/timesplit_no_lig_overlap_train \
--split_val ~/data/splits/timesplit_no_lig_overlap_val \
--log_dir ${save_dir}/workdir \
--wandb_dir ${save_dir} \
--wandb \
--num_dataloader_workers 1 \
--num_workers 1 \
--model_type surface_score_model \
--model_version version${score_version} \
--transformStyle diffdock \
--lr 1e-3 \
--tr_weight 0.33 \
--rot_weight 0.33 \
--tor_weight 0.33 \
--tr_sigma_min 0.1 \
--tr_sigma_max 5 \
--rot_sigma_min 0.03 \
--rot_sigma_max 1.55 \
--batch_size ${batch_size} \
--ns ${ns} \
--nv ${nv} \
--distance_embed_dim 32 \
--cross_distance_embed_dim 32 \
--sigma_embed_dim 32 \
--num_conv_layers ${num_conv_layers} \
--dynamic_max_cross \
--scheduler plateau \
--scale_by_sigma \
--dropout 0.1 \
--remove_hs \
--c_alpha_max_neighbors 24 \
--receptor_radius 15 \
--cudnn_benchmark \
--val_inference_freq 20 \
--num_inference_complexes 500 \
--scheduler_patience 50 \
--n_epochs 2000`
state=$command

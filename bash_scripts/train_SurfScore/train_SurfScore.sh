#!/bin/bash

source ~/anaconda3/bin/activate SurfDock
export TORCH_DISTRIBUTED_DEBUG=INFO
# path of precomputed arrays
export precomputed_arrays="~/precomputed_arrays"
####################################    Setup Parameters    ############################
export CUDA_VISIBLE_DEVICES="6"
main_process_port=29506
num_processes=1
# model version
version_num=6
gussions=20
ns=40
nv=10
num_conv_layers=4
bs_arrays=(64)
valset=2
batch_size=32
mdn_dist_threshold_train=7
mdn_dist_threshold_test=5
lr=1e-3
project="V${version_num}_surface"
time=`date +%Y_%m_%d_%H_%M_%S`
########################################################################################
save_dir="/home/house/caoduanhua/${project}"

for i in ${bs_arrays[@]}
do
batch_size=${i}
echo batch_size : ${batch_size}
command=`accelerate launch \
--multi_gpu \
--main_process_port ${main_process_port} \
--num_processes ${num_processes} \
~/train_mdn_accelarete.py \
--project ${project} \
--run_name project_${project}_valset_${valset}_RunTime_${time}_ns_${ns}_nv_${nv}_layer_${num_conv_layers}_lr_${lr}_gussions_${gussions}_train_dist_${mdn_dist_threshold_train}_test_dist_${mdn_dist_threshold_test}_bs_${batch_size}_use_orig_pos_mdn_inter_loss_atom_type_loss_bond_type_loss \
--test_sigma_intervals \
--data_dir ~/PDBBIND/PDBBind_pocket_8A/ \
--cache_path ~/PDBBIND/cache_RTMScoreFeature_Surface_PDBBIND_pocket_8A_randomsplit_rtmscore_valset_${valset} \
--esm_embeddings_path ~/PDBBIND/esm_embedding/esm_embedding_pocket_for_train/esm2_3billion_embeddings.pt \
--split_test ~/data/splits/timesplit_test \
--split_train ~/data/splits/timesplit_no_lig_overlap_train \
--split_val ~/data/splits/timesplit_no_lig_overlap_val \
--log_dir ${save_dir}/workdir \
--wandb_dir ${save_dir} \
--num_dataloader_workers 1 \
--wandb \
--num_workers 1 \
--lr ${lr} \
--tr_sigma_min 0.000000000000000001 \
--tr_sigma_max 0.00000000000000000001 \
--rot_sigma_min 0.000000000000000001 \
--rot_sigma_max 0.000000000000000000001 \
--tor_sigma_min 0.000000000000000000001 \
--tor_sigma_max 0.00000000000000000000001 \
--no_torsion \
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
--val_inference_freq 5 \
--num_inference_complexes 500 \
--scheduler_patience 30 \
--model_type mdn_model \
--bond_type_prediction \
--atom_type_prediction \
--model_version version${version_num} \
--mdn_dist_threshold_train ${mdn_dist_threshold_train} \
--mdn_dist_threshold_test ${mdn_dist_threshold_test} \
--mdn_dropout 0.1 \
--topN 1 \
--n_gaussians ${gussions} \
--n_epochs 850`
state=$command
done

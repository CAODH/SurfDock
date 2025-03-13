#!/bin/bash
cat << 'EOF'
  ____  _     _ _   _ _   _ ____  _     _____ ____  _   _ ____  _     _     ____  _     ____  _        _ 
  ____              __ ____             _      ____       _         __     __            _             
 / ___| _   _ _ __ / _|  _ \  ___   ___| | __ | __ )  ___| |_ __ _  \ \   / /__ _ __ ___(_) ___  _ __  
 \___ \| | | | '__| |_| | | |/ _ \ / __| |/ / |  _ \ / _ \ __/ _` |  \ \ / / _ \ '__/ __| |/ _ \| '_ \ 
  ___) | |_| | |  |  _| |_| | (_) | (__|   <  | |_) |  __/ || (_| |   \ V /  __/ |  \__ \ | (_) | | | | 
 |____/ \__,_|_|  |_| |____/ \___/ \___|_|\_\ |____/ \___|\__\__,_|    \_/ \___|_|  |___/_|\___/|_| |_| 
                                                                                                       
                                                                                                       
  ____  _     _ _   _ _   _ ____  _     _____ ____  _   _ ____  _     _     ____  _     ____  _        _ 
EOF
                                                                                                       
# This script is used to run SurfDock on test samples
source ~/miniforge3/bin/activate SurfDock
path=$(readlink -f "$0")
SurfDockdir="$(dirname "$(dirname "$(dirname "$path")")")"
SurfDockdir=${SurfDockdir}
echo SurfDockdir : ${SurfDockdir}

temp="$(dirname "$(dirname "$(dirname "$path")")")"
model_temp="$(dirname "$(dirname "$(dirname "$path")")")"

#------------------------------------------------------------------------------------------------#
#------------------------------------ Step0 : Setup Params --------------------------------------#
#------------------------------------------------------------------------------------------------#
echo '------------------------------------ Step1 : Setup Params --------------------------------------'
export precomputed_arrays="${temp}/precomputed/precomputed_arrays"
gpu_string="0"
echo "Using GPU devices: ${gpu_string}"
IFS=',' read -ra gpu_array <<< "$gpu_string"
NUM_GPUS=${#gpu_array[@]}
export CUDA_VISIBLE_DEVICES=${gpu_string}

main_process_port=2951${gpu_array[-1]}
project_name='SurfDock_eval_samples/repeat_250102'
surface_out_dir=${SurfDockdir}/data/eval_sample_dirs/${project_name}/test_samples_8A_surface
data_dir=${SurfDockdir}/data/eval_sample_dirs/test_samples
out_csv_file=${SurfDockdir}/data/eval_sample_dirs/${project_name}/input_csv_files/test_samples.csv
esmbedding_dir=${SurfDockdir}/data/eval_sample_dirs/${project_name}/test_samples_esmbedding
# project_name='SurfDock_Screen_samples/repeat5'

#------------------------------------------------------------------------------------------------#
#----------------------------- Step1 : Compute Target Surface -----------------------------------#
#------------------------------------------------------------------------------------------------#
echo '----------------------------- Step1 : Compute Target Surface -----------------------------------'
mkdir -p $surface_out_dir
cd $surface_out_dir
command=`
python ${SurfDockdir}/comp_surface/prepare_target/computeTargetMesh_test_samples.py \
--data_dir ${data_dir} \
--out_dir ${surface_out_dir} \
`
state=$command

#------------------------------------------------------------------------------------------------#
#--------------------------------  Step2 : Get Input CSV File -----------------------------------#
#------------------------------------------------------------------------------------------------#
echo '--------------------------------  Step2 : Get Input CSV File -----------------------------------'
command=` python \
${SurfDockdir}/inference_utils/construct_csv_input.py \
--data_dir ${data_dir} \
--surface_out_dir ${surface_out_dir} \
--output_csv_file ${out_csv_file} \
`
state=$command

#------------------------------------------------------------------------------------------------#
#--------------------------------  Step3 : Get Pocket ESM Embedding  ----------------------------#
#------------------------------------------------------------------------------------------------#
echo '--------------------------------  Step3 : Get Pocket ESM Embedding  ----------------------------'

esm_dir=${SurfDockdir}/esm
sequence_out_file="${esmbedding_dir}/test_samples.fasta"
protein_pocket_csv=${out_csv_file}
full_protein_esm_embedding_dir="${esmbedding_dir}/esm_embedding_output"
pocket_emb_save_dir="${esmbedding_dir}/esm_embedding_pocket_output"
pocket_emb_save_to_single_file="${esmbedding_dir}/esm_embedding_pocket_output_for_train/esm2_3billion_pdbbind_embeddings.pt"
# get faste  sequence
command=`python ${SurfDockdir}/datasets/esm_embedding_preparation.py \
--out_file ${sequence_out_file} \
--protein_ligand_csv ${protein_pocket_csv}`
state=$command
# esm embedding preprateion

command=`python ${esm_dir}/scripts/extract.py \
"esm2_t33_650M_UR50D" \
${sequence_out_file} \
${full_protein_esm_embedding_dir} \
--repr_layers 33 \
--include "per_tok" \
--truncation_seq_length 4096`
state=$command


# map pocket esm embedding
command=`python ${SurfDockdir}/datasets/get_pocket_embedding.py \
--protein_pocket_csv ${protein_pocket_csv} \
--embeddings_dir ${full_protein_esm_embedding_dir} \
--pocket_emb_save_dir ${pocket_emb_save_dir}`
state=$command

# save pocket esm embedding to single file 
command=`python ${SurfDockdir}/datasets/esm_pocket_embeddings_to_pt.py \
--esm_embeddings_path ${pocket_emb_save_dir} \
--output_path ${pocket_emb_save_to_single_file}`
state=$command


#------------------------------------------------------------------------------------------------#
#------------------------  Step4 : Start Sampling Ligand Confromers  ----------------------------#
#------------------------------------------------------------------------------------------------#
echo '------------------------  Step3 : Start Sampling Ligand Confromers  ----------------------------'

diffusion_model_dir=${model_temp}/model_weights/docking
confidence_model_base_dir=${model_temp}/model_weights/posepredict
protein_embedding=${pocket_emb_save_to_single_file}
test_data_csv=${out_csv_file}

mdn_dist_threshold_test=3.0
version=6
dist_arrays=(3)
for i in ${dist_arrays[@]}
do
mdn_dist_threshold_test=${i}
command=`accelerate launch \
--multi_gpu \
--main_process_port ${main_process_port} \
--num_processes ${NUM_GPUS} \
${SurfDockdir}/inference_accelerate.py \
--data_csv ${test_data_csv} \
--model_dir ${diffusion_model_dir} \
--ckpt best_ema_inference_epoch_model.pt \
--confidence_model_dir ${confidence_model_base_dir} \
--confidence_ckpt best_model.pt \
--save_docking_result \
--mdn_dist_threshold_test ${mdn_dist_threshold_test} \
--esm_embeddings_path ${protein_embedding} \
--run_name ${confidence_model_base_dir}_test_dist_${mdn_dist_threshold_test} \
--project ${project_name} \
--out_dir ${temp}/docking_result/${project_name} \
--batch_size 40 \
--batch_size_molecule 1 \
--samples_per_complex 40 \
--save_docking_result_number 40 \
--head_index  0 \
--tail_index 10000 \
--inference_mode evaluate \
--wandb_dir ${temp}/docking_result/test_workdir`
state=$command
done
cat << 'EOF'
  ____  _     _ _   _ _   _ ____  _     _____ ____  _   _ ____  _     _     ____  _     ____  _        _
  ____              __ ____             _      ____                        _ _               ____                   _  
 / ___| _   _ _ __ / _|  _ \  ___   ___| | __ / ___|  __ _ _ __ ___  _ __ | (_)_ __   __ _  |  _ \  ___  _ __   ___| | 
 \___ \| | | | '__| |_| | | |/ _ \ / __| |/ / \___ \ / _` | '_ ` _ \| '_ \| | | '_ \ / _` | | | | |/ _ \| '_ \ / _ \ | 
  ___) | |_| | |  |  _| |_| | (_) | (__|   <   ___) | (_| | | | | | | |_) | | | | | | (_| | | |_| | (_) | | | |  __/_| 
 |____/ \__,_|_|  |_| |____/ \___/ \___|_|\_\ |____/ \__,_|_| |_| |_| .__/|_|_|_| |_|\__, | |____/ \___/|_| |_|\___(_) 
                                                                    |_|              |___/                             
  ____  _     _ _   _ _   _ ____  _     _____ ____  _   _ ____  _     _     ____  _     ____  _        _
EOF
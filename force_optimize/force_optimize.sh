source ~/anaconda3/bin/activate SurfDock
conda_lib=~/anaconda3/envs/SurfDock/lib
cd ~/SurfDock/force_optimize
export LD_LIBRARY_PATH=$conda_lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="6"
command=`python ./post_energy_minimize.py \
--path_csv csv_path \
--num_process 10 \
`
state=$command
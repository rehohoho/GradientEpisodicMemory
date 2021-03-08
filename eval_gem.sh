export CUDA_VISIBLE_DEVICES=$2
export KMP_DUPLICATE_LIB_OK=TRUE

# memory must be smaller than number of examples per task
python eval.py \
    --n_layers 2 \
    --n_hiddens 100 \
    --data_path data/ \
    --save_path experiment_results/results_nturgbd60_xsub_joint/ \
    --batch_size 16 \
    --log_every 100 \
    --samples_per_task 999999 \
    --data_file $1.pt \
    --cuda yes \
    --seed 0 \
    --model gem \
    --lr 0.05 \
    --n_memories 16 \
    --memory_strength 0.5 \
    --backbone 'MSG3D' \
    --model_args "{num_class: 60, num_point: 25, num_person: 2, num_gcn_scales: 13, num_g3d_scales: 6, graph: graph.ntu_rgb_d.AdjMatrixGraph}" \
    --max_class 60 \
    --checkpoint experiment_results/results_nturgbd60_xsub_joint/gem_nturgbd60_xsub_joint.pt_2021_03_07_18_45_26_fdfa104f2b0841d8b1dba1791d7b00d9.pt

# original config
# --batch_size 10
# --log_every 100
# --samples_per_task 2500
# --cuda yes
# --n_memories 256

# agcn config
# --backbone 'AGCN'
# --model_args "{num_class: 60, num_point: 25, num_person: 2, graph: graph.ntu_rgb_d.Graph, graph_args: {labeling_mode: 'spatial'}}" \

# ms-gcn config
# --backbone 'MS_G3D' \
# --model_args "{num_class: 60, num_point: 25, num_person: 2, num_gcn_scales: 13, num_g3d_scales: 6, graph: graph.ntu_rgb_d.AdjMatrixGraph}" \

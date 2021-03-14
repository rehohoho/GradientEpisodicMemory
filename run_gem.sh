export CUDA_VISIBLE_DEVICES=$2
export KMP_DUPLICATE_LIB_OK=TRUE

cp -r results/ results_$1/

# memory must be smaller than number of examples per task
python main.py \
    --n_layers 2 \
    --n_hiddens 100 \
    --data_path data/ \
    --save_path results_$1/ \
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
    --checkpoint /home/ruien/50c210c_pretrained_without_blocks.pt

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
# data (40091, 3, 300, 25, 2)

# fpha ms-gcn
# --model_args "{num_class: 45, num_point: 21, num_person: 1, num_gcn_scales: 13, num_g3d_scales: 6, graph: graph.ntu_rgb_d.AdjMatrixGraph}" \
# data (600, 3, 300, 21, 2)

cd results_$1/
python plot_results.py
cd ..

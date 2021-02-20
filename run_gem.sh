export CUDA_VISIBLE_DEVICES=0

# memory must be smaller than number of examples per task
python main.py \
    --n_layers 2 \
    --n_hiddens 100 \
    --data_path data/ \
    --save_path results/ \
    --batch_size 1 \
    --log_every 1 \
    --samples_per_task 2 \
    --data_file $1.pt \
    --cuda no \
    --seed 0 \
    --model gem \
    --lr 0.1 \
    --n_memories 2 \
    --memory_strength 0.5 \
    --model_args "{num_class: 60, num_point: 25, num_person: 2, graph: graph.ntu_rgb_d.Graph, graph_args: {labeling_mode: 'spatial'}}"

# original config
# --batch_size 10
# --log_every 100
# --samples_per_task 2500
# --cuda yes
# --n_memories 256

cd results/
python plot_results.py
cd ..
export CUDA_VISIBLE_DEVICES=0
set +o posix
exec > >(tee run_gem_cifar.log) 2>&1

python main.py \
    --n_layers 2 \
    --n_hiddens 100 \
    --data_path data/ \
    --save_path results/ \
    --batch_size 10 \
    --log_every 100 \
    --samples_per_task 2500 \
    --data_file cifar100.pt \
    --cuda yes \
    --seed 0 \
    --model gem \
    --lr 0.1 \
    --n_memories 256 \
    --memory_strength 0.5

cd results/
python plot_results.py
cd ..
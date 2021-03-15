DATA_TYPE1=xview
DATA_TYPE2=joint
BASE_DIR=/home/ruien/pku/${DATA_TYPE1}

python nturgbd60.py \
  --d_tr_path ${BASE_DIR}/train_data_${DATA_TYPE2}.npy \
  --l_tr_path ${BASE_DIR}/train_label.pkl \
  --d_te_path ${BASE_DIR}/val_data_${DATA_TYPE2}.npy \
  --l_te_path ${BASE_DIR}/val_label.pkl \
  --max_class 51 \
  --initial_class_json raw/pkummd_initial_classes_41.json \
  --use_single_task \
  --o pkummd_${DATA_TYPE1}_${DATA_TYPE2}.pt


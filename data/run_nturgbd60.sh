DATA_TYPE1=xview
DATA_TYPE2=joint
BASE_DIR=/home/ltj/codes/MS-G3D/data/ntu_60/${DATA_TYPE1}

python nturgbd60.py \
  --d_tr_path ${BASE_DIR}/train_data_${DATA_TYPE2}.npy \
  --l_tr_path ${BASE_DIR}/train_label.pkl \
  --d_te_path ${BASE_DIR}/val_data_${DATA_TYPE2}.npy \
  --l_te_path ${BASE_DIR}/val_label.pkl \
  --initial_class_json raw/initial_classes_50.json \
  --use_single_task \
  --o nturgbd60_${DATA_TYPE1}_${DATA_TYPE2}.pt


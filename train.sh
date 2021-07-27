export PYTHONWARNINGS="ignore"
export GPU=4

python train.py --config config/aircraft.cfg --gpu ${GPU} --net clar-resnet50 --batch_size 64 --lr 0.015 --eps 0.25
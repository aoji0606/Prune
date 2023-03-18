# Prune For ImageNet

## Train

```shell
# do sparse train
python main.py \
-a resnet50 \
--lr 0.1 \
--epochs 50 \
--batch-size 1024 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--data-path /home/jovyan/fast-data/ \
--save-path ./checkpoints/ \
--sparse-train \
--reg 0.00001

# normal
python main.py \
-a resnet18 \
--lr 0.1 \
--epochs 50 \
--batch-size 1024 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--data-path /home/jovyan/fast-data/ \
--save-path ./checkpoints/
```

## Eval

```shell
python main.py \
-a resnet50 \
--batch-size 100 \
--data-path /home/jovyan/fast-data/ \
--pretrained ./checkpoints/resnet50_best.pth \
--evaluate
```

## Prune

```shell
python main.py \
-a resnet50 \
--lr 0.1 \
--epochs 50 \
--batch-size 1024 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--data-path /home/jovyan/fast-data/ \
--save-path ./checkpoints/ \
--pretrained ./checkpoints/resnet50_best.pth \
--prune \
--prune-rate 0.5
```

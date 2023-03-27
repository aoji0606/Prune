# Prune For ImageNet

## Train

```shell
# do sparse train
python main.py \
-a resnet50 \
--lr 0.1 \
--epochs 50 \
--batch-size 1024 \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--data-path /home/jovyan/fast-data/ \
--save-path ./checkpoints/ \
--dali \
--apex \
--sparse-train \
--reg 0.00001

# normal
python main.py \
-a resnet50 \
--lr 0.1 \
--epochs 50 \
--batch-size 1024 \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--data-path /home/jovyan/fast-data/ \
--save-path ./checkpoints/ \
--dali \
--apex
```

## Eval Normal Model

```shell
python main.py \
-a resnet50 \
--batch-size 100 \
--pretrained ./checkpoints/best_resnet50_checkpoint.pth \
--data-path /home/jovyan/fast-data/ \
--gpu 0 \
--evaluate
```

## Prune

```shell
python main.py \
-a resnet50 \
--lr 0.1 \
--epochs 50 \
--batch-size 1024 \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--pretrained ./checkpoints/best_resnet50_checkpoint.pth \
--data-path /home/jovyan/fast-data/ \
--save-path ./checkpoints/ \
--dali \
--apex \
--prune \
--prune-rate 0.5
```

## Eval Pruned Model

```shell
python main.py \
--pruned-model \
--batch-size 100 \
--pretrained ./checkpoints/best_resnet50_checkpoint_prune0.5.pth \
--data-path /home/jovyan/fast-data/ \
--gpu 0 \
--evaluate
```

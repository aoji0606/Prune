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
--sparse-train \
--reg 0.00001

# normal
python main.py \
-a resnet18 \
--lr 0.1 \
--epochs 50 \
--batch-size 1024 \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--data-path /home/jovyan/fast-data/ \
--save-path ./checkpoints/
```

## Eval Normal Model

```shell
python main.py \
-a resnet50 \
--batch-size 100 \
--data-path /home/jovyan/fast-data/ \
--pretrained ./checkpoints/best_resnet50_checkpoint.pth \
--gpu 0 \
--evaluate

python main.py \
-a resnet18 \
--batch-size 100 \
--data-path /home/jovyan/fast-data/ \
--pretrained ./checkpoints/best_resnet18_checkpoint.pth \
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
--data-path /home/jovyan/fast-data/ \
--save-path ./checkpoints/ \
--pretrained ./checkpoints/best_resnet50_checkpoint.pth \
--prune \
--prune-rate 0.5

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
--pretrained ./checkpoints/best_resnet50_checkpoint.pth \
--prune \
--prune-rate 0.7
```

## Eval Pruned Model

```shell
python main.py \
--pruned-model \
--batch-size 100 \
--data-path /home/jovyan/fast-data/ \
--pretrained ./checkpoints/best_resnet50_checkpoint_prune0.5.pth \
--gpu 0 \
--evaluate

python main.py \
--pruned-model \
--batch-size 100 \
--data-path /home/jovyan/fast-data/ \
--pretrained ./checkpoints/best_resnet50_checkpoint_prune0.7.pth \
--gpu 0 \
--evaluate

```

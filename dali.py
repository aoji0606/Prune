import time
from tqdm import tqdm
import torch
import torch.utils.data
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torchvision.datasets as datasets
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        self.input = ops.FileReader(file_root=data_dir, shard_id=rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.RandomResizedCrop(device="gpu", size=(crop, crop), random_area=[0.08, 1.0])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.images, self.labels = self.input(name="Reader")
        images = self.decode(self.images)
        images = self.resize(images)
        images = self.cmnp(images, mirror=rng)
        return [images, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, val_size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        self.input = ops.FileReader(file_root=data_dir, shard_id=rank, num_shards=world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu", resize_shorter=val_size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.images, self.labels = self.input(name="Reader")
        images = self.decode(self.images)
        images = self.resize(images)
        images = self.cmnp(images)
        return [images, self.labels]


def get_dali_dataloader(type, data_path, batch_size, num_workers, crop, val_size=256):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_workers,
                                    device_id=rank % torch.cuda.device_count(),
                                    data_dir=data_path + '/ILSVRC2012_img_train', crop=crop)
        pip_train.build()
        dataloader = DALIClassificationIterator(pip_train, reader_name="Reader")
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_workers,
                                device_id=rank % torch.cuda.device_count(),
                                data_dir=data_path + '/ILSVRC2012_img_val', crop=crop, val_size=val_size)
        pip_val.build()
        dataloader = DALIClassificationIterator(pip_val, reader_name="Reader")
    else:
        raise ValueError("type error")

    return dataloader


def get_torch_dataloader(type, data_path, batch_size, num_workers, crop, val_size=256):
    if type == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(data_path + '/ILSVRC2012_img_train', transform)
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                                                 num_workers=num_workers, pin_memory=True, sampler=train_sampler)
        return dataloader, train_sampler

    elif type == "val":
        transform = transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(data_path + '/ILSVRC2012_img_val', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                 pin_memory=True)
        return dataloader

    else:
        raise ValueError("type error")


if __name__ == '__main__':
    data_path = "/home/jovyan/fast-data/"

    train_loader = get_dali_dataloader(type='val', data_path=data_path, batch_size=256, num_workers=4, crop=224)
    print('start iterate')
    start = time.time()
    for data in tqdm(train_loader):
        images = data[0]["data"].cuda(non_blocking=True)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
    end = time.time()
    print('end iterate')
    print('dali iterate time: %fs' % (end - start))

    train_loader = get_torch_dataloader(type='val', data_path=data_path, batch_size=256, num_workers=4, crop=224)
    print('start iterate')
    start = time.time()
    for data in tqdm(train_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    print('end iterate')
    print('torch iterate time: %fs' % (end - start))

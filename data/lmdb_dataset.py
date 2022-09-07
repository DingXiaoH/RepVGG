import os
import os.path as osp
from PIL import Image
import six
import lmdb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pyarrow as pa
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

train_lmdb_path = '/apdcephfs/share_1290939/0_public_datasets/imageNet_2012/train.lmdb'
val_lmdb_path = '/apdcephfs/share_1290939/0_public_datasets/imageNet_2012/val.lmdb'

# from data.lmdb_dataset import ImageFolderLMDB, train_lmdb_path, val_lmdb_path
# lmdb_path = train_lmdb_path if is_train else val_lmdb_path
# dataset = ImageFolderLMDB(db_path=lmdb_path, transform=transform)

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getstate__(self):
        state = self.__dict__
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # load label
        target = unpacked[1]
        if self.target_transform is not None:
            target = self.transform(target)

        return img, target
#        if self.transform is not None:
#            img = self.transform(img)
#
#        # im2arr = np.array(img)
#
#        if self.target_transform is not None:
#            target = self.target_transform(target)
#
#        return img, target
        # return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(dpath, name="train", write_frequency=5000):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=4, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()




if __name__ == "__main__":
    # lmdb_path = '/apdcephfs/share_1016399/0_public_datasets/imageNet_2012/train.lmdb'
    # from lmdb_dataset import ImageFolderLMDB
    # dataset = ImageFolderLMDB(db_path=lmdb_path)
    # for x, y in dataset:
    #     print(type(x), type(y))
    # exit()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="The dataset directory to process")
    args = parser.parse_args()
    # generate lmdb
    path = args.dir
    folder2lmdb(path, name="train")
    folder2lmdb(path, name="val")

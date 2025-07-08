from torch.utils.data import Dataset
import sys
import os
from PIL import Image

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root = root
        self.transform = transform
        self.train_dir = os.path.join(self.root, "train")
        self.val_dir = os.path.join(self.root, "val")

        # 创建类别到索引的映射
        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        # 构建图片路径和标签列表
        self._make_dataset(self.Train)

        words_file = os.path.join(self.root, "words.txt")
        wnids_file = os.path.join(self.root, "wnids.txt")

        self.classes = set()
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.classes.add(entry.strip("\n"))

    def _create_class_idx_dict_train(self):
        # 统计训练集类别和图片数量
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        # 统计验证集类别和图片数量
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.val_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        # 构建图片路径和标签的列表
        self.imgs = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_idx[tgt])
                        else:
                            item = (path, self.class_to_idx[self.val_img_to_class[fname]])
                        self.imgs.append(item)
        
        # 保存标签
        self.targets = [tgt for _, tgt in self.imgs]

    def __len__(self):
        # 返回数据集大小
        return self.len_dataset

    def __getitem__(self, idx):
        # 返回图片和标签
        img_path, tgt = self.imgs[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

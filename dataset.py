import os
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import Resize,ToTensor,Compose
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

class MyCifar10(Dataset):
    def __init__(self, root, is_train):
        if is_train:
            data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(root, "test_batch")]
        self.all_images = []
        self.all_labels = []
        for data_file in data_files:
            with open(data_file, "rb") as fo:
                data = pickle.load(fo, encoding="bytes") #load data from pickle file type
                images = data[b'data']
                labels = data[b'labels']
                self.all_images.extend(images)
                self.all_labels.extend(labels)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        image = self.all_images[index]
        image = np.reshape(image, (3, 32, 32)).astype(np.float32)/255.
        label = self.all_labels[index]
        return image, label

# Em chia theo tap Animal_V2 nhe a
class AnimalDataset(Dataset):
    def __init__(self, root, is_train=True,transform = None):
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        if is_train:
            data_path = os.path.join(root, 'train')
        else:
            data_path = os.path.join(root, 'test')

        self.image_paths = []
        self.labels = []

        for i, cat in enumerate(self.categories):#enumerate
            data_cat = os.path.join(data_path, cat)
            for img in os.listdir(data_cat):
                img_path = os.path.join(data_cat, img)
                self.image_paths.append(img_path)
                self.labels.append(i)
        self.transform = transform
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        # image = cv2.imread(self.image_paths[item])
        image = Image.open(self.image_paths[item]).convert("RGB")
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    #chuyen doi dinh dang anh
    transform = Compose([ToTensor(),
        Resize((224, 224))])
    # dataset = MyCifar10(root=)
    dataset = AnimalDataset(root="data/animals", is_train=False,transform=transform)
    # print(image)
    # load data theo tung batch
    dataloader = DataLoader(dataset,
                            batch_size=16,
                            num_workers=8,
                            shuffle=True,
                            drop_last=False)
    # for images, labels in dataloader:
    #     print(images.shape, labels.shape)
    for images, labels in dataloader:
        print(images.shape, labels.shape)
    images, labels = dataloader










import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import model_resnet18 as mymodel

IMAGE_SIZE = 224
NUM_OF_CLASSES = 12
USE_GPU = torch.cuda.is_available()
BATCH_SIZE = 224

CLASSES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
           'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_path_list = self.read_files()
    
    def read_files(self):
        files_path = []
        for file in os.listdir(self.img_dir):
            files_path.append(self.img_dir + "/" + file)
        return files_path
    
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        img_file_name = self.image_path_list[idx]
        image = Image.open(img_file_name)
        image_id = img_file_name.split('/')[-1].split('.')[0]
        
        if self.transform:
            image = self.transform(image)
        
        return [image, image_id]


def gen_kaggle_submission(model, test_data_dir, submission_path):
    data_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_imagesets = TestDataset(test_data_dir, data_test_transform)
    test_dataloaders = DataLoader(test_imagesets, batch_size=BATCH_SIZE, num_workers=4)
    
    predicted_value = {}
    batch_epoch = 0
    for data in test_dataloaders:
        inputs, image_id = data
        
        if USE_GPU:
            inputs = Variable(inputs).cuda()
        else:
            inputs = Variable(inputs)
        
        w = model.features(inputs)
        outputs = model.forward(w)
        _, preds = torch.max(outputs.data, 1)
        
        for img_id, cls in zip(image_id, preds):
            predicted_value[img_id] = cls
        
        batch_epoch += 1
        print("{} image processed!".format(batch_epoch * inputs.size(0)))
    
    np.save('model/predicted-value.npy', predicted_value)
    
    predicted_value = np.load('model/predicted-value.npy').item()
    
    sample_submission = pd.read_csv(submission_path)
    print("submission-column: ", sample_submission.columns.values)
    
    for index, row in (sample_submission.iterrows()):
        img_id = row['file'].split('.')[0]
        
        if img_id in predicted_value:
            sample_submission.set_value(index, 'species', CLASSES[predicted_value[img_id]])
    
    sample_submission.to_csv('model/submission.csv', index=False, header=True)
    print(sample_submission.head())


def load_model(path='model/checkpoint.pth.tar'):
    state = torch.load(path)
    return state


def main():
    model = mymodel.MyResnet(NUM_OF_CLASSES)
    
    state = load_model('model/small-resnet-preconvfet.pth.tar')
    model.load_state_dict(state['state_dict'])
    
    sample_submission_path = 'dataALL/sample_submission.csv'
    test_data_dir = 'dataALL/test'
    
    print("Generating kaggle submission")
    gen_kaggle_submission(model, test_data_dir, sample_submission_path)


if __name__ == "__main__":
    main()

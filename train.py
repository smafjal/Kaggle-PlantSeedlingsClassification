# -*- coding: utf-8 -*-
import copy
import logging
import time

import numpy as np
import torch
from torch.autograd import Variable

import model_resnet18 as mymodel
import utils as utils
import dataloader as mydata

# Logger
logger = logging.getLogger('model-resnet-log')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/mode-resnet-preconvfet.log')
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Paremeters
DATA_ROOT_DIR = 'mydata'
# DATA_ROOT_DIR = 'mydata_small'
MODEL_SAVE_PATH = 'model/small-resnet-preconvfet.pth.tar'
NUM_EPOCHS = 50
BATCH_SIZE = mydata.BATCH_SIZE
USE_GPU = torch.cuda.is_available()
VERBOSE = 5


def generate_batch(conv_features, labels_list):
    labels = np.array(labels_list)
    for idx in range(0, len(conv_features), BATCH_SIZE):
        yield conv_features[idx:min(idx + BATCH_SIZE, len(conv_features))], \
              labels[idx:min(idx + BATCH_SIZE, len(conv_features))]


def batch_train(model, optimizer, criterion, phase, epoch, **kwargs):
    image_datasets = kwargs['image_datasets']
    conv_features = kwargs['conv_features']
    labels_list = kwargs['labels_list']
    
    running_loss = 0.0
    running_corrects = 0
    batch_epoch = 0
    
    for data in generate_batch(conv_features[phase], labels_list[phase]):
        inputs, labels = data
        
        if USE_GPU:
            inputs = Variable(torch.from_numpy(inputs).cuda())
            labels = Variable(torch.from_numpy(labels).cuda())
        else:
            inputs, labels = Variable(torch.from_numpy(inputs)), Variable(torch.from_numpy(labels))
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        outputs = model.forward(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        
        # backward + optimize only if in training phase
        if phase == 'train':
            loss.backward()
            optimizer.step()
        
        running_loss += loss.data[0] * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        batch_epoch += 1
        
        if batch_epoch % VERBOSE == 0:
            logger.info("{} Epoch {} BatchEpoch {}/{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch, batch_epoch,
                len(image_datasets[phase]) // inputs.size(0),
                running_loss / (batch_epoch * inputs.size(0)),
                running_corrects / (batch_epoch * inputs.size(0))))
    
    epoch_loss = running_loss / len(image_datasets[phase])
    epoch_acc = running_corrects / len(image_datasets[phase])
    return epoch_loss, epoch_acc


def train_model(model, **kwargs):
    strt_time = time.time()
    best_model_weight = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.trained_features.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    for epoch in range(1, NUM_EPOCHS):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            
            epoch_loss, epoch_acc = batch_train(model, optimizer, criterion, phase, epoch, **kwargs)
            
            logger.info('{} Epoch {}/{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch, NUM_EPOCHS, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weight = copy.deepcopy(model.state_dict())
        
        logger.info("-" * 20)
    
    time_elapsed = time.time() - strt_time
    logger.info(" ---->Trainig completed! <---")
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_accuracy))
    
    # load best model weights
    model.load_state_dict(best_model_weight)
    
    logger.info("Saving model to %s" % MODEL_SAVE_PATH)
    utils.save_model({
        'epoch': NUM_EPOCHS,
        'state_dict': model.state_dict(),
    }, MODEL_SAVE_PATH)
    
    return model


def main():
    logger.info("Loading data from  %s folder" % DATA_ROOT_DIR)
    dataloaders, image_datasets, class_names = mydata.get_dataloader(DATA_ROOT_DIR)
    
    logger.info("Number of classes found: %s" % len(class_names))
    logger.info("Classes: %s" % class_names)
    
    # logger.info("Viewing some images")
    # utils.show_demo_image(dataloaders)
    
    logger.info("Getting model named resnet18")
    model = mymodel.MyResnet(len(class_names))
    
    conv_features = dict(train=[], val=[])
    labels_list = dict(train=[], val=[])
    
    logger.info("Generating convnet features.......")
    # conv_features['train'], labels_list['train'] = mymodel.preconvfeat(dataloaders, image_datasets, model, phase='train')
    # conv_features['val'], labels_list['val'] = mymodel.preconvfeat(dataloaders, image_datasets, model, phase='val')
    
    logger.info("Saving to disk convnet fetures")
    # utils.save_conv_fetures(conv_features, labels_list, 'model')
    
    logger.info("Load convnet fetures from disk")
    conv_features, labels_list = utils.load_conv_fetures('model')
    
    logger.info("Starting model training..........")
    kwargs = {
        'dataloaders': dataloaders,
        'image_datasets': image_datasets,
        'class_names': class_names,
        'conv_features': conv_features,
        'labels_list': labels_list
    }
    # model = train_model(model, **kwargs)
    
    logger.info("Load model from disk")
    state = utils.load_model(MODEL_SAVE_PATH)
    model.load_state_dict(state['state_dict'])
    
    logger.info("View some image with predicted label")
    # utils.visualize_model_image(model, num_images=6, **kwargs)


if __name__ == "__main__":
    main()

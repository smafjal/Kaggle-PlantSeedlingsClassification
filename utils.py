import numpy as np
import torch
import torchvision
from torch.autograd import Variable

import matplotlib.pyplot as plt

USE_GPU = torch.cuda.is_available()


def plot_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated


def show_demo_image(dataloaders):
    # Get a batch of training data
    inp, classes = next(iter(dataloaders['train']))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inp)
    plot_image(out, title=[[x] for x in classes])


def visualize_model_image(model, num_images=6, **kwargs):
    dataloaders = kwargs['dataloaders']
    class_names = kwargs['class_names']
    
    images_so_far = 0
    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if USE_GPU:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        
        w = model.features(inputs)
        outputs = model.forward(w)
        _, preds = torch.max(outputs.data, 1)
        
        for j in range(inputs.size(0)):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            plot_image(inputs.cpu().data[j])
            
            if images_so_far == num_images:
                plt.savefig('logs/predicted_img.jpg')
                return


def save_model(state, filename='model/checkpoint.pth.tar'):
    torch.save(state, filename)


def load_model(path='model/checkpoint.pth.tar'):
    state = torch.load(path)
    return state


def save_conv_fetures(conv_fetures, labels, root_dir):
    np.save(root_dir + '/conv_feat_train.npy', conv_fetures['train'])
    np.save(root_dir + '/labels_train.npy', labels['train'])
    np.save(root_dir + '/conv_feat_val.npy', conv_fetures['val'])
    np.save(root_dir + '/labels_val.npy', labels['val'])


def load_conv_fetures(_dir='model'):
    conv_fetures = dict(train=[], val=[])
    labels = dict(train=[], val=[])
    
    conv_fetures['train'] = np.load(_dir + '/conv_feat_train.npy')
    labels['train'] = np.load(_dir + '/labels_train.npy')
    
    conv_fetures['val'] = np.load(_dir + '/conv_feat_val.npy')
    labels['val'] = np.load(_dir + '/labels_val.npy')
    return conv_fetures, labels

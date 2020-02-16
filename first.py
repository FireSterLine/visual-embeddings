# -*- coding: utf-8 -*-

# Based on a script written by Sasank Chilamkurthy on BSD licence
# Source: https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py

# The word embedding part was sourced from:
#  Secondary https://github.com/kavgan/nlp-in-practice/blob/master/pre-trained-embeddings/Pre-trained%20embeddings.ipynb

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

def loadDataset(dataset_name = "hymenoptera"):
    print("Loading dataset \"%s\"..." % (dataset_name))
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if dataset_name == "hymenoptera":
        data_dir = 'data/hymenoptera_data'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                     shuffle=True, num_workers=4)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        class_names = image_datasets['train'].classes
    elif dataset_name == "cifar-10":
        data_dir = 'data'
        image_datasets = {x: datasets.CIFAR10(data_dir,
                                              train=(x=='train'),
                                              transform=data_transforms[x],
                                              download=True)
                          for x in ['train', 'val']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                     shuffle=True, num_workers=4)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        class_names = ["plane", "car", "bird", "cat",
                        "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        raise Error("Unknown dataset!")

    print("Datasets sizes: %s" % (str(dataset_sizes)))
    print("# classes = %d ('%s')" % (len(class_names), "', '".join(class_names)))
    print()

    return dataloaders, dataset_sizes, class_names, len(class_names)

######################################################################
# Load embeddings

def loadEmbeddings(model_name = "glove-twitter-25", dataset_name = "hymenoptera"):
    print("Loading embedding model \"%s\"..." % (model_name))

    em_path = model_name+"-"+dataset_name+".kv"
    if not os.path.isfile(em_path):
        if model_name == "GoogleNews-vectors-negative300":
            # Source: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
            # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
            embeddings = KeyedVectors.load_word2vec_format(model_name + '.bin', binary=True).wv
        else:
            embeddings = api.load(model_name).wv
        #embeddings.save(em_path)
        class_embeddings = torch.Tensor(embeddings[class_names])
        torch.save(class_embeddings, em_path)
    else:
        #embeddings = KeyedVectors.load(em_path, mmap='r')
        class_embeddings = torch.load(em_path)
    #class_embeddings = torch.Tensor(embeddings[class_names])

    print("Embeddings shape: %s" % (str(class_embeddings.shape)))
    print()

    return class_embeddings

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                targets = class_embeddings[labels]#.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # print()
                    # print(class_embeddings.shape)
                    # print(targets.shape)
                    # print(outputs.shape)
                    # print(outputs[0].shape)
                    # print(torch.nn.functional.cosine_similarity(outputs[0].unsqueeze_(0), class_embeddings, dim=1))

                    # print()
                    # print(class_embeddings.shape)
                    # print(class_embeddings)
                    #
                    # # print()
                    # # print(outputs.shape)
                    # # print(targets.shape)
                    #
                    # print()
                    # print(outputs)
                    # print(targets)
                    #
                    # print()
                    # print(outputs - targets)
                    #
                    # #print()
                    # #dist = torch.norm(outputs[0] - targets, dim=1, p=None)
                    # #print(dist)

                    preds = torch.LongTensor([torch.norm(output - class_embeddings, dim=1, p=None).topk(1, largest=False)[1] for output in outputs])
                    # preds = torch.LongTensor([torch.nn.functional.cosine_similarity(output.unsqueeze_(0), class_embeddings, dim=1).topk(1, largest=False)[1] for output in outputs])

                    # print(torch.norm(outputs[0] - class_embeddings, dim=1, p=None))
                    # print(torch.norm(outputs[1] - class_embeddings, dim=1, p=None))
                    # print(torch.norm(outputs[2] - class_embeddings, dim=1, p=None))
                    # print(torch.norm(outputs[3] - class_embeddings, dim=1, p=None))
                    #
                    # print(preds.shape)
                    # print(labels.data.shape)

                    # loss = torch.norm(outputs - targets, dim=1, p=None)
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

######################################################################
# Finetuning the convnet
# ----------------------
#

# dataset = "hymenoptera"
dataset = "cifar-10"

embeddings = "glove-twitter-25"
# embeddings = "GoogleNews-vectors-negative300"
######################################################################


print()

dataloaders, dataset_sizes, class_names, n_classes = loadDataset(dataset)

class_embeddings = loadEmbeddings(embeddings,dataset)

print(zip(class_names,class_embeddings))


model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features



model_ft.fc = nn.Linear(num_ftrs, class_embeddings.shape[1])

# TODO: Add a second last fully-connected layer
# intermed = (n_classes+num_ftrs)//2
# model_ft.fc = nn.Sequential(
#     nn.Linear(num_ftrs, intermed)
#     , embedding_layer
# )

# This allows to set the learning rate for different layers
par_lrs = {
    'firstlayer'   :  {'params' : [], 'lr': 0.0001},
    'scnlastlayer' :  {'params' : [], 'lr': 0.0001},
    'lastlayers'   :  {'params' : [], 'lr': 0.001},
    'others'       :  {'params' : [], 'lr': 0.0},
}

for name,param in model_ft.named_parameters():
    print(name)
    if name.startswith("conv1."):
        par_lrs["firstlayer"]['params'].append(param)
    elif name.startswith("layer4.1"):
        par_lrs["scnlastlayer"]['params'].append(param)
    elif name.startswith("fc"):
        par_lrs["lastlayers"]['params'].append(param)
    else:
        par_lrs["others"]['params'].append(param)
print()

for i in par_lrs:
    print(i + ":\t" + str(len(par_lrs[i]['params'])) + '\t' + str(par_lrs[i]['lr']))
    if par_lrs[i]['lr'] == 0.0:
        for param in par_lrs[i]["params"]:
            param.requires_grad = False
print()


model_ft = model_ft.to(device)

# TODO Use that geometric softmax idea!
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# criterion = nn.CosineSimilarity()

# 0. optimize only laste layer's parameter
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 1. optimize layers according to the settings (see a few lines above)
# TODO: switch back to SGD
# optimizer_ft = optim.SGD([par_lrs[i] for i in par_lrs if par_lrs[i]['lr'] != 0.0],
#                      lr = 0.001,
#                      momentum = 0.9
#                      # , weight_decay = 0.9
#                  )
optimizer_ft = optim.Adam([par_lrs[i] for i in par_lrs if par_lrs[i]['lr'] != 0.0],
                     lr = 0.001
                     # , weight_decay = 0.9
                 )

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

######################################################################
#

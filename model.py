# -*- coding: utf-8 -*-
import copy
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

# Logger
logger = logging.getLogger('model-resnet-log')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('mode-resnet-preconvfet.log')
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# my DATA_ROOT_DIR was 'mydata' which structure is:
# mydata
#     | -- train
#           | -- cats
#           | -- dogs
#     | -- val
#           | -- cats
#           | -- dogs
#     | -- test

# Paremeters
DATA_ROOT_DIR = 'mydata'
MODEL_SAVE_PATH = 'resnet-preconvfet.pth.tar'
USE_GPU = torch.cuda.is_available()
IMAGE_SIZE = 224
BATCH_SIZE = 20
NUM_EPOCHS = 20
NUM_OF_CLASSES = 2
VERBOSE = 5


def get_dataloader(data_root_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_root_dir, x), data_transforms[x]) for x in
                      ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    return dataloaders, image_datasets, image_datasets['train'].classes


def save_model(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_model(path='checkpoint.pth.tar'):
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
        inputs = inputs.view(inputs.size(0), -1)

        # forward
        outputs = model.fc(inputs)
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


def train_model(model, optimizer, criterion, scheduler, num_epochs=5, **kwargs):
    strt_time = time.time()
    best_model_weight = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(1, num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            epoch_loss, epoch_acc = batch_train(model, optimizer, criterion, phase, epoch, **kwargs)

            logger.info('{} Epoch {}/{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch, num_epochs, epoch_loss, epoch_acc))

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
    save_model({
        'epoch': num_epochs,
        'state_dict': model.state_dict(),
    }, MODEL_SAVE_PATH)

    return model


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
    for data in test_dataloaders:
        inputs, image_id = data

        if USE_GPU:
            inputs = Variable(inputs).cuda()
        else:
            inputs = Variable(inputs)

        # resnet-init-block
        w = model.conv1(inputs)
        w = model.bn1(w)
        w = model.relu(w)
        w = model.maxpool(w)

        # layer1-4
        w = model.layer1(w)
        w = model.layer2(w)
        w = model.layer3(w)
        w = model.layer4(w)

        w = model.avgpool(w)
        w = w.view(inputs.size(0), -1)
        # forward
        outputs = model.fc(w)
        _, preds = torch.max(outputs.data, 1)
        for img_id, cls in zip(image_id, preds):
            predicted_value[img_id] = cls

    sample_submission = pd.read_csv(submission_path)
    for index, row in (sample_submission.iterrows()):
        img_id = str(row['id'])
        if img_id in predicted_value:
            sample_submission.set_value(index, 'label', predicted_value[img_id])

    saved_path='model/my_submission.csv'
    sample_submission.to_csv(saved_path)
    print("Submission file saved on",saved_path)
    

def get_model(num_of_class):
    model_resnet = torchvision.models.resnet18(pretrained=True)
    for param in model_resnet.parameters():
        param.requires_grad = False

    # num_ftrs = model_resnet.fc.in_features
    model_resnet.fc.out_features = num_of_class
    for param in model_resnet.fc.parameters():
        param.requires_grad = True

    if USE_GPU:
        model_resnet = model_resnet.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_resnet = torch.optim.SGD(model_resnet.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_resnet, step_size=7, gamma=0.1)

    return model_resnet, optimizer_resnet, criterion, step_lr_scheduler


def preconvfeat(dataloaders, image_datasets, model, phase):
    conv_fet = []
    labels_list = []

    logger.info("ConvNet fetures generating phase: %s" % phase)
    batch_epoch = 0
    for data in dataloaders[phase]:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # resnet-init-block
        w = model.conv1(inputs)
        w = model.bn1(w)
        w = model.relu(w)
        w = model.maxpool(w)

        # layer1-4
        w = model.layer1(w)
        w = model.layer2(w)
        w = model.layer3(w)
        w = model.layer4(w)

        # avgpool & fc
        w = model.avgpool(w)

        conv_fet.extend(w.data.numpy())
        labels_list.extend(labels.data.numpy())
        batch_epoch += 1
        if batch_epoch % VERBOSE == 0:
            logger.info("Gen CovNet Feture! {} Epoch {}/{}".format(
                phase, batch_epoch, len(image_datasets[phase]) // inputs.size(0)))

    conv_fet = np.concatenate([[feat] for feat in conv_fet])
    return conv_fet, labels_list


def main():
    logger.info("Loading data from  %s folder" % DATA_ROOT_DIR)
    dataloaders, image_datasets, class_names = get_dataloader(DATA_ROOT_DIR)

    logger.info("Number of classes found: %s" % len(class_names))
    logger.debug("Classes: %s" % class_names)

    logger.info("Getting model named resnet18")
    model, optimizer, criterion, lr_scheduler = get_model(len(class_names))

    print model

    # conv_features = dict(train=[], val=[])
    # labels_list = dict(train=[], val=[])

    # logger.info("Generating convnet features.......")
    # conv_features['train'], labels_list['train'] = preconvfeat(dataloaders, image_datasets, model, phase='train')

    # conv_features['val'], labels_list['val'] = preconvfeat(dataloaders, image_datasets, model, phase='val')

    # logger.info("Saving to disk convnet fetures")
    # save_conv_fetures(conv_features, labels_list, 'model')

    # logger.info("Load convnet fetures from disk")
    # conv_features, labels_list = load_conv_fetures('model')

    # logger.info("Starting model training..........")
    # kwargs = {
    #     'dataloaders': dataloaders,
    #     'image_datasets': image_datasets,
    #     'class_names': class_names,
    #     'conv_features': conv_features,
    #     'labels_list': labels_list
    # }
    # model = train_model(model, optimizer, criterion, lr_scheduler, num_epochs=NUM_EPOCHS, **kwargs)

    # logger.info("Load model from disk")
    # state = load_model('resnet-preconvfet.pth.tar')
    # model.load_state_dict(state['state_dict'])

    # sample_submission_path = 'sample_submission.csv'
    # test_data_dir = 'mydata/test'
    # gen_kaggle_submission(model, test_data_dir, sample_submission_path)


if __name__ == "__main__":
    main()


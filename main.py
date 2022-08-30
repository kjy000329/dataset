import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from function import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import sys
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping
from torch.utils.data.sampler import SubsetRandomSampler

data_path = r'Tactile_slip/D20382'


def get_data(label_file):
    img_paths = []
    labels = []
    csv = pd.read_csv(label_file, header=0)  # ===0000????
    csv = csv.sample(frac=1.0)  # csv是一个数据框，csv.loc[index]是一个series，.value得到array
    for index in csv.index:
        image = str(csv.loc[index].values[0])
        label = str(csv.loc[index].values[1])
        image_name = os.path.join(data_path, image)
        img_paths.append(image_name)
        labels.append(label)
    return img_paths, labels


def create_datasets(batch_size):
    # Detect devices
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # file paths

    label_file = "Tactile_slip/slip.csv"
    save_model_path = "./Conv3D_ckpt/"

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    img_paths, labels = get_data(label_file)

    le = LabelEncoder()
    le.fit(labels)
    list(le.classes_)

    all_X_list = img_paths
    all_y_list = labels2cat(le, labels)

    # percentage of training set to be used for validation
    validation_size = 0.2

    # train, test split
    train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.2,
                                                                      random_state=0)

    # image transformation
    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    train_set, test_set = Dataset_3DCNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                          Dataset_3DCNN(data_path, test_list, test_label, selected_frames, transform=transform)

    # split into training and validation batches

    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)

    valid_sampler = SubsetRandomSampler(valid_idx)

    # loading train, validation and test data

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              num_workers=0)

    # create model
    cnn3d = CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y,
                  drop_p=dropout, fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2, num_classes=k).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn3d = nn.DataParallel(cnn3d)

    optimizer = torch.optim.Adam(cnn3d.parameters(), lr=learning_rate)  # optimize all cnn parameters

    return train_loader, test_loader, valid_loader


def train(model, batch_size, patience, epochs):
    # train
    # set model as training mode
    model.train()
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    N_count = 0  # counting total trained sample in one epoch

    patience = 20
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):

        for batch_idx, (X, y) in enumerate(train_loader):
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            N_count += X.size(0)
            optimizer.zero_grad()
            output = model(X)
            # print('train output shape', output.shape)
            loss = criterion(output, y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # validate the model
        model.eval()
        for (X, y) in valid_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            output = model(X)
            loss = criterion(output, y)
            valid_losses.append(loss.item())  # sum up batch loss

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model, avg_train_losses, avg_valid_losses


model = CNN3D()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# use CPU for running
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# setting the random seeds
random_state = 0
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# 3D CNN layers and dropout parameters
fc_hidden1, fc_hidden2 = 20, 10
dropout = 0

# training parameters (k = no. target category)
k = 2
log_interval = 10
img_x, img_y = 240, 320
epochs = 100
batch_size = 4
patience = 10
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
learning_rate = 0.001

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1
selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

# create CNN3D model
cnn3d = CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y,
              drop_p=dropout, fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2, num_classes=k).to(device)

train_loader, test_loader, valid_loader = create_datasets(batch_size)
model, train_loss, valid_loss = train(model, batch_size, patience, epochs)

# visualize the loss as the network trained
fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss)) + 1
plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1)  # consistent scale
plt.xlim(0, len(train_loss) + 1)  # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')

"Test our trained model"
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))

model.eval()  # prep model for evaluation

for batch, (data, target) in enumerate(test_loader):
    # data, target = data.to(device), target.to(device).view(-1, )
    target1 = target.data[:, 0]
    if len(target1.data) != batch_size:
        break
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target1)
    # update test loss
    test_loss += loss.item() * data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target1.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target1.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss / len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(5):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

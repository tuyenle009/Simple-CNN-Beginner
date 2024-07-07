import torch
import torch.nn as nn
from dataset import AnimalDataset
import argparse
#/arpars/
from torch.utils.data import DataLoader
from models import SimpleCNN
from torchvision.models import resnet34, ResNet34_Weights, vgg16, VGG16_Weights
from torchvision.transforms import ToTensor,Compose,Resize,Normalize, RandomAffine, ColorJitter
from tqdm.autonotebook import tqdm #import then you can use on jupyternotebook? collap
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import shutil
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Pastel1") #change colors here
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    #built arguments
    parser = argparse.ArgumentParser(description="train NN model")
    #add argument
    parser.add_argument("--data_path", "-d", type=str, default="data/animals", help="path to the dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--lr", "-l", type=float, default=1e-2)
    parser.add_argument("--log_path", "-p", type=str, default="tensorboard/animals")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/animals")
    args = parser.parse_args()
    return args

def train(args):
    #create device to access GPU and model, epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #if torch.cuda.is_available() else "cpu"
    # model = SimpleCNN().to(device) #turn on gpu = implace function
    # print(torch.cuda.is_availabhtople())
    # model = resnet34(weights=ResNet34_Weights.DEFAULT)
    # print(model)

    model = vgg16(pretrained=True)
    model_name='vgg16'
    if model_name == 'vgg16':
        model = vgg16(pretrained=True)
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features
        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 10), nn.LogSoftmax(dim=1))
    if model_name == 'resnet34':
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(in_features=4096, out_features=10) #4096 512
        # frozen parameters
        for param in model.parameters():
            param.requires_grad= False
        for param in model.fc.parameters():
            param.requires_grad = True

    model.to(device)
    # transform data
    train_transform = Compose([
        #cac gia tri se co toa do ngau nhien
        RandomAffine(#nhung so duoi day do anh viet hoc qua nhieu nen mac dinh thanh
            degrees=(-5,5), #or 5 |
            translate= (0.15,0.15), #dich trai dich phai 15%buc anh
            scale=(0.85,1.15),# giu nguyen or thu nho 85% anh or phong to 115% anh
            shear=10
        ),
        ColorJitter(brightness=0.125,contrast=0.5, saturation=0.5, hue=0.05), #doi mau buc anh
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]
    ) #normalize images
    test_transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ]) #normalize images
    #TRAIN
    train_dataset = AnimalDataset(root=args.data_path, is_train=True, transform=train_transform)
    # dataloader
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        num_workers = 8,
        shuffle = True,
        drop_last = False)
    # VALIDATION
    test_dataset = AnimalDataset(root="data/animals", is_train=False, transform=test_transform)
    # dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=False,
        drop_last=False)
    #criterion ( /kraɪˈtɪəriən/ )~standard, optimizer
    criterion= nn.CrossEntropyLoss()
    optimizer= torch.optim.SGD(model.parameters(),lr=args.lr)
    #write loss in tensorboard in process train and test, inthe future we can observe
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    #ghi lại quá trình để lần tiếp theo train có thể biết được quá trình mình train như nào
    #write the processing, in the future i can continue to train model
    writer = SummaryWriter(args.log_path)
    best_acc=-1
    epochs = args.epochs
    #built training processing
    for epoch in range(epochs):
        # process bar
        progress_bar = tqdm(train_dataloader, colour="cyan")
        #TRAIN
        model.train()
        for iteration, (images, labels) in enumerate(progress_bar): #16 images
            #take output from model
            images = images.to(device)
            labels = labels.to(device)
            output = model(images) #qua cac layer gradient tu dong duoc luu tru
            loss = criterion(output,labels)#tinh ra loss = pp crossentropy
            #su dung progress bar: epoch/epochs and iter/len(train_da) loss
            progress_bar.set_description("TRAIN | epoch:{}/{} | Iter{}/{} | loss:{:0.4f}".format(epoch,epochs,iteration,len(train_dataloader),loss))
            #writting in dir (Loss/train)
            writer.add_scalar("Loss/train",loss,epoch*len(train_dataloader)+iteration)
            #updata weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #VALIDATION
        all_losses = []
        all_labels = []
        all_predictions = []
        model.eval()  # BachNorm Dropout != process train model
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_dataloader): #16 images
                #take output from model
                images = images.to(device)
                labels = labels.to(device)
                output = model(images) #qua cac layer gradient tu dong duoc luu tru
                loss = criterion(output,labels)#tinh ra loss = pp crossentropy. | type pytorch
                #lay ra cac gia tri du doan tao thành thành 1 vector voi torch.argmax
                # _,prediction = torch.max(output,dim=1)
                predictions = torch.argmax(output, dim=1)
                #add all elements into list -loss, labels, predictions
                all_losses.append(loss.item())
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
            #cacular mean loss
            loss = np.mean(all_losses)
            #built accuracy_score
            accuracy = accuracy_score(all_labels,all_predictions)
            #built conf_matrix
            conf_matrix= confusion_matrix(all_labels,all_predictions)
            print("TEST | epoch:{}/{} | Loss:{:0.4f} | Acc:{:0.4f}".format(epoch + 1, epochs, loss,accuracy))
            #add_scalar test - loss, accuracy
            writer.add_scalar("Test/Loss",loss,epoch)
            writer.add_scalar("Test/Accuracy",accuracy,epoch)
            #add confusion matrix into writer
            plot_confusion_matrix(writer, conf_matrix, train_dataset.categories, epoch)

            checkpoint = {
                "model_state_dict": model.state_dict(), #all parameter of model, not include architecture
                "epoch": epoch, #we can know we have trained numbers of epoch
                "optimizer_state_dict": optimizer.state_dict() #we can see path which we went to
            }
            #save checkpoint last because we can continue to train in the future
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
            if accuracy > best_acc: #save checkpoint best because we can deploy in fact
                torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))
                best_acc = accuracy
if __name__ == '__main__':
    args = get_args()
    train(args)


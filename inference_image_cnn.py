import torch
import torch.nn as nn
import argparse
from torchvision.models import resnet34, ResNet34_Weights
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
import cv2
def get_args():
    #built arguments
    parser = argparse.ArgumentParser(description="train NN model")
    #add argument
    parser.add_argument("--image_path", "-p", type=str, default="data/Others/4.jpg")#required= True)#default="data/Others/4.jpg"
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/animals")
    args = parser.parse_args()
    return args

def inference(args):

    classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    #create device to access GPU and model, epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #if torch.cuda.is_available() else "cpu"
    #read images
    image = cv2.imread(args.image_path)
    # cv2.imshow("dog",image)
    # cv2.waitKey()
    print(image.shape)
    image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
    image = cv2.resize(image,(args.image_size, args.image_size))
    #mean and standard of imagenet
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]
    #equivalent to Tensor() from Pytorch
    image = image/255.
    #equivalent to Normalize
    image = (image - mean)/std
    image = np.transpose(image,(2,0,1))[None,:,:,:] #add batch size
    image = torch.from_numpy(image).float().to(device) #convert image to tensor
    #take a model
    model = resnet34()
    model.fc = nn.Linear(in_features=512,out_features=10)
    #load a best model
    checkpoint = torch.load(os.path.join(args.checkpoint_path,"best.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        #check level believe of model with class
        prob = softmax(output)
        predicted_prob,predicted_class = torch.max(prob,dim=1) #probability
        print(predicted_prob[0].item())
        score = predicted_prob[0].item()*100
        cv2.imshow("{} with confident score of {}%".format(classes[predicted_class[0].item()],score),cv2.imread(args.image_path))
        cv2.waitKey(0)

if __name__ == '__main__':
    args = get_args()
    inference(args)


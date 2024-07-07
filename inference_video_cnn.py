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
    parser.add_argument("--video_path", "-p", type=str, default="data/Others/cat.mp4")
    parser.add_argument("--output_path", "-o", type=str, default="data/Others/output.mp4")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/animals")
    args = parser.parse_args()
    return args

def inference(args):
    #classes
    classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    #create device to access GPU and model, epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #if torch.cuda.is_available() else "cpu"
    model = resnet34()
    model.fc = nn.Linear(in_features=512, out_features=10)
    #take the best model - load_state_dict
    checkpoint = torch.load(os.path.join(args.checkpoint_path, "best.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    #VIDEO - VideoCapture
    cap = cv2.VideoCapture(args.video_path) #open video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height of video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #width of video
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*"MJPG"),
                          int(cap.get(cv2.CAP_PROP_FPS)),(width, height))

    softmax = nn.Softmax()
    while cap.isOpened(): #khi nao video van mo thi ta doc vao frame
        flag, frame = cap.read() #flag =1 khi frame loi ex: mat pixel
        if not flag: #khi nao loi thi out ra (khong khong loi = co loi)
            break
        #read images
        # frame = cv2.imread(args.video_path)
        image = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
        image = cv2.resize(image,(args.image_size, args.image_size))
        #mean and standard of imagenet
        mean=[0.485,0.456,0.406]
        std=[0.229,0.224,0.225]
        #equivalent to Tensor() from Pytorch
        image = image/255.
        #equivalent to Normalize
        image = (image - mean)/std
        image = np.transpose(image,(2,0,1))[None,:,:,:] #add batch size
        image = torch.from_numpy(image).float().to(device)
        #
        with torch.no_grad():
            output = model(image)
            #check level believe of model with class
            prob = softmax(output)
            predicted_prob,predicted_class = torch.max(prob,dim=1)
            print(predicted_prob[0].item())
            score = predicted_prob[0].item()*100
            frame = cv2.putText(frame, "{} - {:0.2f}%".format(classes[predicted_class[0].item()],score), (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX ,1, (0,0,255), 2, cv2.LINE_AA)
            out.write(frame)
    # the video capture and video
    # write objects
    out.release()
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_args()
    inference(args)


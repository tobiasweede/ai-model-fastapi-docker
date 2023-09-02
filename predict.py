#! /usr/bin/env python
"""
Module that consists of function that will help to predict the output given an image.

convert_img_to_grey_tensor -> function that receive a image (PIL) transform into grey, 
and finally to a tensor to be used by the model_predict

model_predict -> responsible for predict the image and returns a number

"""

import torch
from PIL import Image
from torchvision.transforms import transforms

from architecture import create_cnn_model

WEIGHTS_PATH = "weights/mnist_net.pth"


def convert_img_to_grey_tensor(img: Image.Image):
    grey_img = img.convert("L")
    transform = transforms.Compose([transforms.PILToTensor()])
    image_tensor = transform(grey_img)
    return image_tensor


def model_predict(img: Image.Image):
    net = create_cnn_model()
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device("cpu")))
    print("Model Loaded!")
    image_tensor = convert_img_to_grey_tensor(img).type("torch.FloatTensor")
    outputs = net(torch.unsqueeze(image_tensor, 0))
    _, pred = torch.max(input=outputs, dim=1)
    return pred.item()

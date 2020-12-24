import numpy as np
import cv2

import torch

# import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
# from PIL import Image, ImageDraw
# from Ipython import display
import torchvision
from torchvision import transforms
from torch import nn
# import torch.nn.functional as F
import torchvision

model_path = "D:/Computer vision/Face Mask Detection/face_mask_path.pth"


# class MobileFaceMask(nn.Module):
#     def __init__(self):
#         super(MobileFaceMask, self).__init__()
#
#         self.network = torchvision.models.mobilenet_v2(pretrained=True)
#         for child in self.network.children():
#             for param in child.parameters():
#                 param.requires_grad = True
#         self.network.classifier = torch.nn.Sequential(nn.Linear(1280, 256),
#                                                       nn.ReLU(),
#                                                       nn.Dropout(p=0.25),
#                                                       nn.Linear(256, 2),
#                                                       nn.Sigmoid())
#
#     def _slow_forward(self, x):
#         x = self.network(x)
#         return x


model = torchvision.models.mobilenet_v2(pretrained=True)
classifier = nn.Sequential(nn.Linear(1280,256),nn.ReLU(),nn.Dropout(0.25),nn.Linear(256,2),nn.Sigmoid())
model.fc = classifier

model.load_state_dict(torch.load(model_path))

trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device {device}")

labels = ["Mask", "No Mask"]
labelColor = [(10, 255, 0), (10, 0, 255)]

cap = cv2.VideoCapture(0)

# MTCNN for detecting the presence of faces
mtcnn = MTCNN(keep_all=True, device=device)

model.to(device)

model.eval()
while True:
    ret, frame = cap.read()

    if ret == False:
        pass

    img_ = frame.copy()
    boxes, _ = mtcnn.detect(img_)  # We saved box coordinates in boxes and left the landmarks
    try:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            x1, y1 = max(x1,0), max(y1, 0)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face = img_[int(y1-30):int(y2+30), int(x1-30):int(x2+30)]

            in_img = trans(face)
            in_img = in_img.unsqueeze(0)
            in_img = in_img.to(device)

            out = model(in_img)
            prob = torch.exp(out)
            a = list(prob.squeeze())
            predicted = a.index(max(a))
            textSize = cv2.getTextSize(labels[predicted], cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)[0]
            textX = x1 + (x2-x1)//2-textSize[0]//2
            cv2.putText(frame, labels[predicted], (int(textX), y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, labelColor[predicted], 2)
    except (TypeError, ValueError) as e:
        pass

    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()

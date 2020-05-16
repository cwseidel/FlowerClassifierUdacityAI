
import torch
from torchvision import models

#model = models.vgg11(pretrained=True)
#print(model.classifier[0].in_features)
model = models.vgg11_bn(pretrained=True)
print("vgg11_bn", model.classifier[0].in_features)
model = models.vgg13(pretrained=True)
print("vgg13", model.classifier[0].in_features)
model = models.vgg13_bn(pretrained=True)
print("vgg13_bn", model.classifier[0].in_features)
model = models.vgg16(pretrained=True)
print("vgg16", model.classifier[0].in_features)
model = models.vgg16_bn(pretrained=True)
print("vgg16_bn", model.classifier[0].in_features)

model = models.densenet121(pretrained=True)
print("densenet121", model.classifier.in_features)
model = models.densenet161(pretrained=True)
print("densenet161", model.classifier.in_features)



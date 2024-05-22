import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms

def train_object_detection_model():

    model = fasterrcnn_resnet50_fpn(pretrained=True)

    for name, param in model.named_parameters():
        param.requires_grad = False

    torch.save(model.state_dict(), 'object_detection_model.pth')

if __name__ == "__main__":
    train_object_detection_model()

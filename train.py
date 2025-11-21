import torch
import torchvision
from torchvision import transforms
from datasets.coco_dataset import CustomCOCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from utils import engine
import io
from contextlib import redirect_stdout

DIR_TEST = "input_images"
root = "runs/labelme2coco"
train = "runs/labelme2coco/train.json"
val = "runs/labelme2coco/val.json"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PATH = "weights/model.pth"

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model_instance_segmentation(num_classes):

    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    return transforms.Compose(custom_transforms)

if __name__ == '__main__':
    
    with redirect_stdout(io.StringIO()):
        train_dataset = CustomCOCO(root, train, get_transform())
        test_dataset = CustomCOCO(root, val, get_transform())
        
    train_batch_size = 1

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)

    num_classes = len(train_dataset.classes) + 1
    
    num_epochs = 40
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=num_epochs + 25,
        T_mult=1,
        verbose=True
    )
    len_dataloader = len(train_dataloader)
    
    for epoch in range(num_epochs):
        engine.train_one_epoch(model=model, optimizer=optimizer, data_loader=train_dataloader, device=device, epoch=epoch, print_freq=10)
        lr_scheduler.step()
        engine.evaluate(model=model, data_loader=test_dataloader, device=device)
        
    torch.save(model.state_dict(), PATH)
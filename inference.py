import os
from re import L
import cv2
import numpy as np
import glob
import torch
import torchvision
from torchvision import transforms
from datasets.coco_dataset import CustomCOCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from utils import custom_utils, engine
from utils.predictor import VisualizationDemo
import io
from contextlib import redirect_stdout

from detectron2.config import get_cfg

from ultra.model.model import parsingNet
from ultra.utils.common import merge_config
from ultra.utils.dist_utils import dist_print
import scipy.special, tqdm
import torchvision.transforms as ts
from ultra.data.dataset import LaneTestDataset
from ultra.data.constant import culane_row_anchor, tusimple_row_anchor


import warnings
warnings.filterwarnings("ignore")

cfg = get_cfg()
cfg.merge_from_file("Configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
cfg.MODEL.WEIGHTS = "weights/model_final_c10459.pkl"
cfg.MODEL.DEVICE = 'cpu'

demo = VisualizationDemo(cfg)

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

DIR_TEST = "input_images"
root = "runs/labelme2coco"
train = "runs/labelme2coco/train.json"
val = "runs/labelme2coco/val.json"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PATH = "weights/model.pth"

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ ==  '__main__':
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
    class_names = {}
    for cat in train_dataloader.dataset.coco.dataset['categories']:
        class_names[cat['id']] = cat['name']

    signClasses = []
    personClasses = ['Walking', 'Standing', 'Biking', 'Sitting', 'Running', 'Motorbiking']
    lightClasses = ['Green_light', 'Red_light', 'Yellow_light']
    for name in class_names.values():
        if name not in personClasses and name not in lightClasses:
            signClasses.append(name)
    

    # Similar Colors per category
    COLORS = np.random.uniform(0, 255, size=(len(class_names)))
    c = []
    cSign = []
    cLight = []
    cPed = []
    for i in range(80):
        c.append(255)

    for i in range(len(signClasses)):
        cSign.append([255, COLORS[i], COLORS[i]])
    for i in range(len(personClasses)):
        cPed.append([COLORS[i], 255, COLORS[i]])
    for i in range(len(lightClasses)):
        cLight.append([COLORS[i], COLORS[i], 255])

    num_epochs = 40
    model = get_model_instance_segmentation(num_classes)

    ULTRA_BACKBONE = '50'
    ULTRA_GRINDING_NUM = 100
    ULTRA_cls_num_per_lane = 56
    ULTRA_img_w, ULTRA_img_h = 2048,1024
    ULTRA_row_anchor = tusimple_row_anchor

    net = parsingNet(pretrained = False, backbone=ULTRA_BACKBONE,cls_dim = (ULTRA_GRINDING_NUM+1,ULTRA_cls_num_per_lane,4),
                    use_aux=False) # we dont need auxiliary segmentation in testing
    net.to(device)
    #PATH TO TEST_MODEL WHICH IS WEIGHTS
    ULTRA_state_dict = torch.load('weights/ep030.pth', map_location='cpu')['model']
    ULTRA_compatible_state_dict = {}
    for k, v in ULTRA_state_dict.items():
        if 'module.' in k:
            ULTRA_compatible_state_dict[k[7:]] = v
        else:
            ULTRA_compatible_state_dict[k] = v

    ULTRA_img_transforms = ts.Compose([
        ts.Resize((288, 800)),
        ts.ToTensor(),
        ts.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # move model to the right device
    model.to(device)

    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    # print("Overall Accuracy", correct/total)
    # for category in class_correct.keys():
    #     accuracy = class_correct[category] / class_total[category]
    #     print(f"{category.capitalize()} Accuracy:", accuracy)
    
    test_images = glob.glob(f"{DIR_TEST}/*.png")
    for i in range(len(test_images)):
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        img = Image.open(test_images[i])
        image = cv2.imread(test_images[i])

        # Detectron2 predictions
        detectron2_outputs, visualized_detectron2_output = demo.run_on_image(image)
        visualized_detectron2_output.save(f"inference_outputs/images/{image_name}_detectron2.jpg")

        orig_image = cv2.imread(f"inference_outputs/images/{image_name}_detectron2.jpg")
        
        ultra_img = ULTRA_img_transforms(img)
        ultra_img = torch.unsqueeze(ultra_img, 0)
        
        print(ultra_img.shape)
        with torch.no_grad():
            out = net(ultra_img.to(device))

        col_sample = np.linspace(0, 800 - 1, ULTRA_GRINDING_NUM)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(ULTRA_GRINDING_NUM) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == ULTRA_GRINDING_NUM] = 0
        out_j = loc

        
        for j in range(out_j.shape[1]):
            if np.sum(out_j[:, j] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, j] > 0:
                        ppp = (int(out_j[k, j] * col_sample_w * ULTRA_img_w / 800) - 1, int(ULTRA_img_h * (ULTRA_row_anchor[ULTRA_cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(orig_image, ppp,5,(255,255,0),-1)

        
        # BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply transforms
        image_input = get_transform()(image)
        # Add batch dimension.
        image_input = torch.unsqueeze(image_input, 0)
        # Predictions

        with torch.no_grad():
            outputs = model(image_input.to(device))

        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        print(outputs)

        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:

            # draw the bounding boxes and write the class name on top of it

            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # Filter out boxes according to `detection_threshold`.
            boxes = boxes[scores >= 0.6].astype(np.int32)
            draw_boxes = boxes.copy()
            # Get all the predicited class names.
            pred_classes = [class_names[i-1] for i in outputs[0]['labels'].cpu().numpy()]
            
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]

                color = []
                if class_name in signClasses:
                    color = cSign[signClasses.index(class_name)]
                elif class_name in personClasses:
                    color = cPed[personClasses.index(class_name)]
                elif class_name in lightClasses:
                    color = cLight[lightClasses.index(class_name)]
                else:
                    color = COLORS[list(class_names.values()).index(class_name)]

                orig_image = custom_utils.draw_boxes(orig_image, box, color, None)
                orig_image = custom_utils.put_class_text(
                    orig_image, box, class_name,
                    color, None
                )

            #cv2.waitKey(1)
            cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)

print('TEST PREDICTIONS COMPLETE')
        

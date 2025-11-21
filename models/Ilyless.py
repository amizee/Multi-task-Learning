import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from functools import partial
from torchvision.models.resnet import resnet50, ResNet50_Weights
from backbones.backbone_utils_changed import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

import torchvision.transforms as ts
import torch
from torch import nn, Tensor
from models.FasterRCNNBackless import fasterrcnn_resnet50_fpn1, FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
from models.UltraBackless import parsingNet

def create_model(num_classes: int, rcnnWeightPath: str, ultraWeightPath: str):
    trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
    
    model = IylessNet(
        rcnn_num_classes=num_classes,
        rcnnWeightPath=rcnnWeightPath,
        ultraWeightPath = ultraWeightPath
    )

    return model

class IylessNet(nn.Module):
    def __init__(self, rcnn_num_classes, ultraWeightPath=None, rcnnWeightPath = None):
        super().__init__()
        
        #rcnn

        self.rcnn_num_classes = rcnn_num_classes
        self.rcnnHead = fasterrcnn_resnet50_fpn1(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            pretrained=True
        )
        
        # get the number of input features 
        in_features = self.rcnnHead.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        self.rcnnHead.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes = self.rcnn_num_classes) 

        if rcnnWeightPath is not None:
            self.rcnnHead.load_state_dict(torch.load(rcnnWeightPath, map_location=torch.device('cpu')), strict=False)

        
        self.backbone = self.rcnnHead.backbone
        #ultra

        ULTRA_GRINDING_NUM = 100
        ULTRA_cls_num_per_lane = 56

        self.ultraHead = parsingNet(self.backbone, cls_dim = (ULTRA_GRINDING_NUM+1,ULTRA_cls_num_per_lane, 4), use_aux=False)
        
        ULTRA_state_dict = torch.load(ultraWeightPath, map_location='cpu')['model']
        ULTRA_compatible_state_dict = {}
        for k, v in ULTRA_state_dict.items():
            if 'module.' in k:
                ULTRA_compatible_state_dict[k[7:]] = v
            else:
                ULTRA_compatible_state_dict[k] = v

        self.ultraHead.load_state_dict(ULTRA_compatible_state_dict, strict=False)


    

        

    def forward(self, images, targets=None):

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        
        out, features = self.backbone(images)
        
        print([(k, v.shape) for k, v in out.items()])

        layers = [value for key, value in out.items()]
        x2, x3, x4 = layers[1], layers[2], layers[3]
        
        ultraOutput = self.ultraHead(x2,x3,x4)
        rcnnOutput = self.rcnnHead(features, images, targets, original_image_sizes)
        
        
        return ultraOutput, rcnnOutput
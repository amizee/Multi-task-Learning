import torch
import torchvision
from torchvision import transforms
from datasets.coco_dataset import CustomCOCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
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

def calculate_iou(box1, box2):
    # Get coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the areas of the two bounding boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the IoU
    iou = intersection_area / (area1 + area2 - intersection_area)

    return iou


if __name__ == '__main__':
    
    with redirect_stdout(io.StringIO()):
        test_dataset = CustomCOCO(root, val, get_transform())
        
    train_batch_size = 1

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)

    num_classes = len(test_dataset.classes) + 1
    class_names = {}
    for cat in test_dataloader.dataset.coco.dataset['categories']:
        class_names[cat['id']] = cat['name']

    signClasses = []
    personClasses = ['Walking', 'Standing', 'Biking', 'Sitting', 'Running', 'Motorbiking']
    lightClasses = ['Green_light', 'Red_light', 'Yellow_light']
    for name in class_names.values():
        if name not in personClasses and name not in lightClasses:
            signClasses.append(name)
    
    num_epochs = 40
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    class_correct = {'sign': 0, 'light': 0, 'person': 0}
    class_total = {'sign': 0, 'light': 0, 'person': 0}

    # Confusion matrix: TP, FP, FN
    task_confusion_matrix = {'sign': [0, 0, 0],
                                'light': [0, 0, 0],
                                'person': [0, 0, 0],}
    num_gt_task = {'sign': 0,
                                'light': 0,
                                'person': 0,}
    for batch in test_dataloader:
        images, annos = batch
        gt_boxes = annos[0]['boxes']
        gt_labels = annos[0]['labels']
        #print("GT boxes", gt_boxes)
        #print("GT labels", gt_labels)

        images = tuple(image.to(device) for image in images)
        with torch.no_grad():
            outputs = model(images)
        
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
      
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].data.numpy()
        pred_boxes = boxes[scores >= 0.6]
        pred_labels = labels[scores >= 0.6]
        total += len(gt_labels)
        # print('total: ', total)
        # print(pred_labels)
        # print(pred_boxes)
        # print("Number of predictions:", len(pred_labels))

        # Counting GT labels per task per image

        
        class_correct_per_image = {'sign': 0, 'light': 0, 'person': 0}

        gt_labels_list = gt_labels.tolist()
        for i in range(len(gt_labels)):
            class_name = class_names[gt_labels_list[i]]
            for category in num_gt_task.keys():
                if class_name in globals()[f"{category}Classes"]:
                    num_gt_task[category] += 1
                    break
        
        num_correct = 0
        for i in range(len(pred_boxes)):
            fp_flag = True   
            task_name = '' 
            class_name = class_names[pred_labels[i]]
            for category in class_correct.keys():
                if class_name in globals()[f"{category}Classes"]:
                    class_total[category] += 1
                    task_name = category
                    break
            
            if pred_labels[i] not in gt_labels:
                # False positive
                task_confusion_matrix[task_name][1] += 1
                continue
            for j in range(len(gt_labels)):
                if pred_labels[i] == gt_labels[j]:
                    iou = calculate_iou(pred_boxes[i], gt_boxes[j])
                    if iou > 0.5:
                        fp_flag = False
                        class_name = class_names[pred_labels[i]]
                        if class_name in globals()[f"{category}Classes"]:
                            # If predicted class has IOU > 0.5, and is predicted correctly
                            class_correct[category] += 1
                            class_correct_per_image[category] += 1

                            # Increment True Positive
                            task_confusion_matrix[task_name][0] += 1
                        correct += 1
                        # num_correct += 1
                        break
                    
            if fp_flag == True:
                # False positive
                task_confusion_matrix[task_name][1] += 1
        
        # print("Number of correct predictions for image", num_correct)
    
    task_confusion_matrix['light'][2] = num_gt_task['light'] - class_correct_per_image['light']
    task_confusion_matrix['person'][2] = num_gt_task['person'] - class_correct_per_image['person']
    task_confusion_matrix['sign'][2] = num_gt_task['sign'] - class_correct_per_image['sign']

    # TP, FP, FN
    print('Confusion matrix: ', task_confusion_matrix)

    overall_tp = task_confusion_matrix['light'][0] + task_confusion_matrix['person'][0] + task_confusion_matrix['sign'][0]
    overall_fp = task_confusion_matrix['light'][1] + task_confusion_matrix['person'][1] + task_confusion_matrix['sign'][1]
    overall_fn = task_confusion_matrix['light'][2] + task_confusion_matrix['person'][2] + task_confusion_matrix['sign'][2]
    
    overall_precision = (overall_tp)/(overall_tp + overall_fp)
    light_precision = task_confusion_matrix['light'][0]/(task_confusion_matrix['light'][0] + task_confusion_matrix['light'][1])
    person_precision = task_confusion_matrix['person'][0]/(task_confusion_matrix['person'][0] + task_confusion_matrix['person'][1])
    sign_precision = task_confusion_matrix['sign'][0]/(task_confusion_matrix['sign'][0] + task_confusion_matrix['sign'][1])

    overall_recall = (overall_tp) / (overall_tp + overall_fn)
    light_recall = (task_confusion_matrix['light'][0]) / (task_confusion_matrix['light'][0] + task_confusion_matrix['light'][2])
    sign_recall = (task_confusion_matrix['sign'][0]) / (task_confusion_matrix['sign'][0] + task_confusion_matrix['sign'][2])
    person_recall = (task_confusion_matrix['person'][0]) / (task_confusion_matrix['person'][0] + task_confusion_matrix['person'][2])
    
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)
    light_f1 = (2 * light_precision * light_recall) / (light_precision + light_recall)
    sign_f1 = (2 * sign_precision * sign_recall) / (sign_precision + sign_recall)
    person_f1 = (2 * person_precision * person_recall) / (person_precision + person_recall)

    num_lights_total = num_gt_task['light']
    num_sign_total = num_gt_task['sign']
    num_person_total = num_gt_task['person']

    accuracy = correct/total
    light_accuracy = class_correct['light'] / num_lights_total
    sign_accuracy = class_correct['sign'] / num_sign_total
    person_accuracy = class_correct['person'] / num_person_total
    
    print('==========================================================\n',
          f'Overall Metrics: \nPrecision: {overall_precision}\nAccuracy: {accuracy}\nRecall: {overall_recall}\nF1: {overall_f1}\n'
          '==========================================================\n'
          f'Light Metrics: \nPrecision: {light_precision}\nAccuracy: {light_accuracy}\nRecall: {light_recall}\nF1: {light_f1}\n'
          '==========================================================\n'
          f'Sign Metrics: \nPrecision: {sign_precision}\nAccuracy: {sign_accuracy}\nRecall: {sign_recall}\nF1: {sign_f1}\n'
          '==========================================================\n'
          f'Person Metrics: \nPrecision: {person_precision}\nAccuracy: {person_recall}\nRecall: {person_recall}\nF1: {person_f1}\n'
          '==========================================================\n'
          f'Number of Objects per category: \nLights: {num_lights_total}\nSigns: {num_sign_total}\nPerson: {num_person_total}\n'
          '==========================================================\n')
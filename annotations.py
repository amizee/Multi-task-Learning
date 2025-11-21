import os
from collections import Counter
from datasets.coco_dataset import CustomCOCO

image_root = "runs/labelme2coco"
annotation_file = "runs/labelme2coco/train.json"

dataset = CustomCOCO(image_root, annotation_file)
class_ids = [ann["category_id"] for ann in dataset.coco.dataset["annotations"]]
class_counts = Counter(class_ids)
class_names = dataset.coco.getCatIds()
category_distribution = {dataset.idx_to_name(class_id): class_counts[class_id] for class_id in class_names}

signClasses = []
personClasses = ['Walking', 'Standing', 'Biking', 'Sitting', 'Running', 'Motorbiking']
lightClasses = ['Green_light', 'Red_light', 'Yellow_light']

signCount = 0
personCount = 0
lightCount = 0

num_categories = len(dataset.coco.dataset['categories'])
print("Number of Categories in Training Dataset: {}\n".format(num_categories))

sorted_category_distribution = dict(sorted(category_distribution.items(), key=lambda item: item[1], reverse=True))
print("Category Distribution:")
for category, count in sorted_category_distribution.items():
    if category in personClasses:
        personCount += count
    elif category in lightClasses:
        lightCount += count
    else:
        signCount += count
    
    print("{}: {}".format(category, count))
print()

print("Number of Sign Labels: {}".format(signCount))
print("Number of Person Labels: {}".format(personCount))
print("Number of Light Labels: {}".format(lightCount))
print("Total Number of Labels in Training Dataset: {}\n".format(sum(sorted_category_distribution.values())))

num_images = len(dataset)
print("Number of Images in Training Dataset: {}".format(num_images))


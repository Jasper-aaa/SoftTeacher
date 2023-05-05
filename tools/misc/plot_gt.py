import os
from pycocotools.coco import COCO
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

json_path = "/home/lab/YYF/coco/annotations/instances_train2017.json"
img_path = "/home/lab/YYF/coco/train2017"

coco = COCO(annotation_file=json_path)
ids = list(sorted(coco.imgs.keys()))

coco_cls = dict([ (v["id"],v["name"] )for k,v in coco.cats.items()])

for img_id in ids[205:210]:
    ann_ids = coco.getAnnIds(imgIds=img_id)

    targets = coco.loadAnns(ann_ids)

    img_file_name = coco.loadImgs(img_id)[0]['file_name']

    img = Image.open(os.path.join(img_path,img_file_name)).convert('RGB')
    draw = ImageDraw.Draw(img)

    for target in targets:
        x,y,w,h = target["bbox"]
        x1,y1,x2,y2= x,y, int(x+w),int(y+h)
        draw.rectangle((x1,y1,x2,y2))

    ax = plt.axes()
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    plt.imshow(img)
    plt.show()
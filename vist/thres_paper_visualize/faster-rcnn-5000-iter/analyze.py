import pickle

with open("f-rcnn-5000.pkl","rb") as f:
    data = pickle.load(f)

def count_num_detection_per_img(thres:float,result:list):
    total_num = 0
    filter_num = 0
    for detection in result:
        num_detection = detection.shape[0]
        total_num += num_detection
        if num_detection > 0:
            for i in range(num_detection):
                if detection[i][4]>thres:
                    filter_num += 1
    return total_num,filter_num

val_1 = data[0]
total,filter = count_num_detection_per_img(0.3,val_1)
thres = [i/10 for i in range(10)]
def calulate_thres(thres,data:list):
    filter_detect_num_list = []
    for thr in thres:
        total_detect_num = 0
        filter_detect_num = 0
        for img in data:
            total_num,filter_num = count_num_detection_per_img(thr,img)
            total_detect_num+=total_num
            filter_detect_num+=filter_num
        filter_detect_num_list.append(filter_detect_num)
    return total_detect_num,filter_detect_num_list

total_detect_num,filter_detect_num_list = calulate_thres(thres,data)

print(total_detect_num,filter_detect_num_list)


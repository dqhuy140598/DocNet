from pretrained.detection.craft import CRAFT
from pretrained.detection import imgproc
from pretrained.detection import craft_utils
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config.config import DETECTION_CONFIG
class TextDetection:

    def __init__(self,model_path,text_threshold=0.7,low_text=0.4,link_threshold=0.4,cuda=False,canvas_size=1280,mag_ratio=1.5,poly=False,show_time=False):
        self.model_path = model_path
        self.text_threshold= text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.cuda = cuda
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        self.show_time = show_time
        self.create_net()

    def __copyStateDict(self,state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def __preprocess_image(self,img):
        if img.shape[0] == 2: img = img[0]
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:   img = img[:, :, :3]
        img = np.array(img)

        return img

    def __test_net(self,image):

        t0 = time.time()

        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        y, feature = self.net(x)

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)

        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if self.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text


    def __draw_result(self,img, boxes):
        print(boxes[0])
        for i, box in enumerate(boxes):
            copy = img.copy()
            x1,y1,x2,y2 = box
            roi = copy[y1:y2,x1:x2]
            # cv2.imwrite('out/{0}.jpg'.format(i),roi)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        plt.imshow(img)
        plt.show()

    def create_net(self):
        self.net = CRAFT()
        print('Loading weights from checkpoint (' + self.model_path + ')')
        if self.cuda:
            self.net.load_state_dict(self.__copyStateDict(torch.load(self.model_path)))
        else:
            self.net.load_state_dict(self.__copyStateDict(torch.load(self.model_path, map_location='cpu')))

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

    def __str2bool(self,v):
        return v.lower() in ("yes", "y", "true", "t", "1")

    def detect_text_in_image(self,image):
        self.net.eval()
        image = self.__preprocess_image(image)
        bboxes, polys, score_text = self.__test_net(image)
        bboxes = self.__convert_to_rectangle(bboxes)
        # self.__draw_result(image, bboxes)
        result = self.__line_segmentation(bboxes)
        # self.__draw_result(image,result['8'])
        return result

    def __convert_to_rectangle(self,bboxes):
        recs = []
        for bbox in bboxes:
            x1,y1,x2,y2 = int(bbox[0][0]),int(bbox[0][1]),int(bbox[2][0]),int(bbox[2][1])
            recs.append([x1,y1,x2,y2])

        return recs

    def __line_segmentation(self,bboxes):

        bboxes.sort(key=lambda x: x[1])
        result = {}
        line = 0
        result[str(line)] = []
        # print(bboxes)
        for i in range(0,len(bboxes)-1):
            if abs(bboxes[i][1] - bboxes[i+1][1]) >= abs(bboxes[i][1] - bboxes[i][3]) -2:
                result[str(line)].append(bboxes[i])
                result[str(line)].sort(key=lambda x:x[0])
                line +=1
                result[str(line)] = []
            else:
                result[str(line)].append(bboxes[i])

        result[str(line)].append(bboxes[-1])

        for key,line in result.items():
            line.sort(key=lambda x:x[0])
        #
        # for key,line in result.items():
        #     box_min_y = min(line,key= lambda x:x[1])
        #     box_max_y = max(line,key=lambda x:x[3])
        #     for box in line:
        #         box[1] = box_min_y[1]
        #         box[3] = box_max_y[3]
        return result

if __name__ == '__main__':

    detection = TextDetection(DETECTION_CONFIG.MODEL_PATH,DETECTION_CONFIG.TEXT_THRESHOLD,DETECTION_CONFIG.LOW_TEXT,\
                              DETECTION_CONFIG.LINK_THRESHOLD,DETECTION_CONFIG.CUDA,DETECTION_CONFIG.CANVAS_SIZE,\
                              DETECTION_CONFIG.MAG_RATIO,DETECTION_CONFIG.POLY,DETECTION_CONFIG.SHOW_TIME)

    image_path = 'test/test5.jpg'
    image = cv2.imread(image_path)
    result = detection.detect_text_in_image(image)
    print(result)



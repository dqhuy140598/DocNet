from detection import TextDetection
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import string
from pretrained.recognition.utils import CTCLabelConverter,AttnLabelConverter
from pretrained.recognition.dataset import AlignCollate
from pretrained.recognition.model import Model
from data.dataset import MyDataset
from config.config import DETECTION_CONFIG,RECOGNITION_CONFIG
import cv2
import silx
class TextRecognition:

    def __init__(self,text_detection,opt):

        self.opt = opt

        if opt.sensitive:
            self.opt.character = string.printable[:-6]

        cudnn.benchmark = True
        cudnn.deterministic = True
        self.opt.num_gpu = torch.cuda.device_count()

        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character)
        else:
            self.converter = AttnLabelConverter(self.opt.character)

        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3

        self.model = Model(self.opt)

        self.device = torch.device('cuda:0')

        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction,opt.CUDA)

        self.model = torch.nn.DataParallel(self.model).to(self.device)

        print('loading pretrained model from %s' % opt.saved_model)

        self.model.load_state_dict(torch.load(opt.saved_model,map_location = self.device))

        self.AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

        self.text_detection = text_detection

    def recognize_text(self,image):

        copy = image.copy()
        result = self.text_detection.detect_text_in_image(image)

        demo_data = MyDataset(image=image, list_dict=result)
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self.AlignCollate_demo, pin_memory=True)

        self.model.eval()
        lines = []
        with torch.no_grad():
            text_document = ''
            for i,(image_tensors, image_path_list) in enumerate(demo_loader):
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)

                # For max length prediction
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

                if 'CTC' in self.opt.Prediction:
                    preds = self.model(image, text_for_pred).log_softmax(2)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.permute(1, 0, 2).max(2)
                    preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                    preds_str = self.converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = self.model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)

                print(len(preds_str))

                line_pred = []

                for pre in preds_str:
                    word = pre[:pre.find('[s]')]
                    line_pred.append(word)


                text = " ".join(line_pred)
                if i == 0:
                    text_document = text
                else:
                    text_document = text_document + "\n"+ text

                del image

            return text_document


if __name__ == '__main__':
    detection = TextDetection(DETECTION_CONFIG.MODEL_PATH, DETECTION_CONFIG.TEXT_THRESHOLD, DETECTION_CONFIG.LOW_TEXT, \
                              DETECTION_CONFIG.LINK_THRESHOLD, DETECTION_CONFIG.CUDA, DETECTION_CONFIG.CANVAS_SIZE, \
                              DETECTION_CONFIG.MAG_RATIO, DETECTION_CONFIG.POLY, DETECTION_CONFIG.SHOW_TIME)

    recognition = TextRecognition(detection,RECOGNITION_CONFIG)
    image_path = 'test/test3.jpg'
    image = cv2.imread(image_path)
    text_document = recognition.recognize_text(image)
    print(text_document)







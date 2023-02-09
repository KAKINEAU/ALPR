import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


#################### personal modification
# reading palte number 
import easyocr
import csv 
import re 
import difflib
import pytesseract

import numpy as np
def four_point_transform(image, tl, tr, br, bl):
    print("dans four point")
    print("topleft", tl)
    print("topr", tr)
    print("botr", br)
    print("botl", bl)
	# obtain a consistent order of the points and unpack them
	# individually
	#rect = order_points(pts)
	#(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    cv2.imshow("warped",warped)
    print("cv2imshow")
	# return the warped image
    return warped



dt_save = []
img_save = []
Last_img_data = []

### Read plate tesseract ocr
def read_plate(image_path):
    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY) # turn image into black and white

    gray = cv2.GaussianBlur(gray, (3,3), 0) # apply GaussianBlur filter to eliminate noise 

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # path of tesseract.exe 

    text = pytesseract.image_to_string(gray)
    return text, image_path

### Read plate easy ocr
def read_license_plate(image_path):
    reader = easyocr.Reader(['fr'])
    result = reader.readtext(image_path, paragraph="False", allowlist= "ABCDEFGHJKLMNPQRSTUWXYZ0123456789")
    return result[0][1], image_path # return license plate text and the image 

### compare two string
def similarity(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).ratio()

### filter format of the plate AA-000-AA doesn't work yet
def filter_plate(licence_text):
    print("filter")    
    if re.match("^[A-Z]{2}-[0-9]{3}-[A-Z]{2}$", licence_text):
        print("valid format ")
    else :
        for index, ch in enumerate(licence_text):
            if index <= 1 or index >=5 and not re.match("^[A-Z]$", licence_text) :
                #print(index, ch)
                licence_texte= licence_text.replace("*",licence_text[index] )
                print (licence_texte)
            #if index > 1 or index <5 and not re.match("^[0-9]$", licence_text) :
              #  print(index, ch)
             #   re.sub("[A-Z]","*",licence_text)
    return licence_text
#######

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    global save_dir
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        crop_rate = 30   # perform OCR every ... frames

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #make crop folder
                #if not os.path.exists("crop"):
                #        os.mkdir("crop")
                
                # Write results
                for *xyxy, conf, cls in reversed(det):

                    ############# personal modif
                    if opt.crop :                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                        if frame % crop_rate ==0 :
                            try:
                                print("We execute our modifications (OCR, ...) \n")
                                #crop an image based on coordinates
                                object_coordinates = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]

                                
                                
                                print("object coordinates : ",object_coordinates)
                                cropobj = im0[int(xyxy[1])-5:int(xyxy[3])+5,int(xyxy[0])-5:int(xyxy[2])+5]

                                cropobj2 = im0[int(xyxy[1])-15:int(xyxy[3])+15,int(xyxy[0])-15:int(xyxy[2])+15]
                                cv2.imwrite("crop2.jpg",cropobj2)
                                #print("find-platepoints): ",find_plate_points(cropobj))
                                xmin = int(xyxy[0]-5)
                                ymin =int(xyxy[1]-5)
                                xmax =int(xyxy[2]+5)
                                ymax =int(xyxy[3]+5)
                                tl = xmin, ymin
                                tr = xmax, ymin
                                br = xmax, ymax
                                bl = xmin, ymax
                                rect = [tl,tr,br,bl]
                                
                                four_point_transform(cropobj,tl,tr,br,bl)
                                print("xyxy object coordinates : ",xyxy)
                                try :
                                    # perform the OCR
                                    ocr_result, img_path = read_license_plate(cropobj)  # easyOCR
                                    #ocr_result, img_path = read_plate(cropobj)  # Tesseract
                                    print("ocr_result : ", ocr_result,"\n")
                                    
                                    # save and get the previous OCR result if exist
                                    if len(dt_save) != 0:  # check if list not empty
                                        temp_last = dt_save[-1]  # get previous ocr text
                                        img_last = img_save[-1]  # get previous image

                                    dt_save.append(ocr_result)  
                                    img_save.append(img_path)
                                    #current_last = dt_save[-1]

                                    # get the ratio similarity between the two last detection (see if it's the same plate)
                                    similarity_ratio = similarity(temp_last, ocr_result)
                                    print("similarity between (temp,current) :", temp_last, ocr_result," is  =", similarity_ratio)

                                    if similarity_ratio ==1 or (len(dt_save)>=5 and similarity_ratio >=0.7):
                                        dt_save.clear()
                                        all_detection = [item[1] for item in Last_img_data]
                                        for i in all_detection:
                                            print("ratio entre ", i , "et", ocr_result)
                                            if similarity(i, ocr_result) >= 0.7 : # licence plate is already in the list
                                                print('ratio >0.7 license plate already in list ')
                                                pass  # send a message for the robot to move
                                    else:
                                        # if not in a list we save it (DEFAULT)
                                        crop_file_path = os.path.join(save_dir, str(time.strftime("%Y%m%d-%H%M%S"))+".jpg")
                                        Last_img_data.append([crop_file_path,ocr_result,img_last]) # save each photos with (path(Date) + ocr result +img save )
                                except :
                                    pass
                            except :
                                    pass
                    ############# personal modif        


                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    
    #################### personal modification
    parser.add_argument('--crop', action='store_true', help='crop dectection')
    #parser.add_argument('--ocr', action='store_true', help='ocr text reading dectection')
    ####################
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7tiny_custom.pt']:
            #for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

            ### save image in the folder
            print("liste final des elements : ", Last_img_data)
            for data_image in Last_img_data:
                print("path",data_image[0], "image", data_image[-1])
                cv2.imwrite(data_image[0],data_image[-1]) #save the crop .jpg file   cv2.imwrite(path, image coordinate)

            ### save all data  in a csv (image name , detection result, GPS info, )
            csv_file_path =os.path.join(save_dir, "Detection_results.csv")
            with open(csv_file_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                writer.writerow(['image name', 'ocr detection'])
                for item in Last_img_data:
                    writer.writerow([os.path.basename(item[0]),item[1]])
            
            ### compare our database detection to the data of our research
            #csv1 with csv 2
            
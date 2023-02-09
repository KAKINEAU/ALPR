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
from image_processing_fcts import similarity, read_license_plate, image_processing, read_plate, order_points
# reading palte number 
#import easyocr
import csv 
import re 
import difflib
#import pytesseract
import threading

def filter_plate2(licence_text):
    pattern = re.compile("[A-Z]{2}-[0-9]{3}-[A-Z]{2}") #|[A-Z]{2}[0-9]{3}[A-Z]{2}
    result = re.search(pattern,licence_text)
    
    if result:
        #print("\033[32m plate is {} => result is {} \033[0m".format(licence_text, result.group()))
        
        return result.group()
    return None

def save_image(image,filename):
        #print("THREAD image save")
        cv2.imwrite(filename, image)


def filter(filename):
    
    #print("\nFIlter\n")
    image = cv2.imread(filename)
    if image is None or image.size == 0:
        print("Error: Could not read image")
    else :
        warped = image_processing(image)
        cv2.imwrite(filename, warped)
        print("\033[34mwarped\033[0m")

def read_plate_tesseract(filename):
    
    global ocr_result
    image = cv2.imread(filename)
    ocr_result = read_plate(image,filename)
    if ocr_result != "":
        print("read_plate tesseract", ocr_result)

Last_img_data = []
"""    
dt_save = []
img_save = []
def process_ocr_result(ocr_text, image_path):
    #print("PROCESS OCR RESULT",len(dt_save))

    if len(dt_save) != 0:
        temp_last = dt_save[-1]
        img_last = img_save[-1]
        print("temp_last",temp_last)
    else:
        temp_last = []
        img_last = [] 
    dt_save.append(ocr_text)
    img_save.append(image_path)

    similarity_ratio = similarity(temp_last, ocr_text) if len(dt_save) != 0 else 0
    
    if similarity_ratio == 1 or (len(dt_save) >= 5 and similarity_ratio >= 0.7):
        dt_save.clear()
        all_detection = [item[1] for item in Last_img_data]
        for i in all_detection:
            if similarity(i, ocr_text) >= 0.7:
                print("{} déjà dans la liste on peut donc supprimer l'image".format(ocr_text))
                return None
    else:
        #crop_file_path = os.path.join(save_dir, str(time.strftime("%Y%m%d-%H%M%S")) + ".jpg")
        Last_img_data.append([image_path, ocr_text])
        return Last_img_data
"""
def check_plate_uniqueness(ocr_text, image_path):
    print("check_plate_uniqueness",len(Last_img_data))
    all_detections = [item[1] for item in Last_img_data]

    if all_detections and any(similarity(i, ocr_text) >= 0.7 for i in all_detections):
        return None
    else:
        Last_img_data.append([image_path, ocr_text])
        return Last_img_data
    
def get_rightmost_plate_coords(boxes_coords, plate_number):
    max_x1 = float('-inf')
    rightmost_plate = None
    for i in range(plate_number):
        x1 = (boxes_coords[i][0]).item()
        if x1 > max_x1:
             max_x1 = x1
             rightmost_plate = boxes_coords[i]
    return rightmost_plate
#################### personal modification

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

        crop_rate = 60   # perform OCR every ... frames

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
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
    
                    ############# personal modif
                    if opt.crop :                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                        if frame % crop_rate ==0 :
                            try:
                                # Handle multiple plates
                                num_of_detected_plates = len(det)
                                print("il y a {} plaque(s)".format(num_of_detected_plates))

                                rightmost_plate_coords = get_rightmost_plate_coords(det[:, :4],num_of_detected_plates)
                                print(rightmost_plate_coords)
                                # Extract detection square coords top-left(x1,y1), bottom-right(x2,y2)
                                x1 = (rightmost_plate_coords[0]).item()
                                y1 = (rightmost_plate_coords[1]).item()
                                x2 = (rightmost_plate_coords[2]).item()
                                y2 = (rightmost_plate_coords[3]).item()
                                
                                # Crop inside the image using square coords
                                cropobj = im0[int(y1)-10:int(y2)+10,int(x1)-10:int(x2)+10]

                                #create a path for the image (dir : last_run folder, name : YYYYMMDD-HHMMSS.jpg)
                                image_path = os.path.join(save_dir,str(time.strftime("%Y%m%d-%H%M%S"))+".jpg")
                                
                                # Save image
                                thread = threading.Thread(target=save_image, args=(cropobj,image_path))
                                thread.start()
                                thread.join() # Attendre que le thread soit terminé
                                
                                # Image Processing function to warp the image
                                thread2 = threading.Thread(target=filter, args=(image_path,))
                                thread2.start()
                                thread2.join() # Attendre que le thread soit terminé
                                
                                # OCR Processing function to extract text
                                thread3 = threading.Thread(target=read_plate_tesseract, args=(image_path,))
                                thread3.start()
                                thread3.join() # Attendre que le thread soit terminé

                                if not ocr_result:
                                    raise ValueError ("OCR result is empty => next image")

                                # See the format of the detection
                                valid_format = filter_plate2(ocr_result)
                                if valid_format != None:
                                    if check_plate_uniqueness(valid_format, image_path) == None:
                                        raise ValueError ("Already in the list".format(valid_format))
                                    else :
                                        print("\033[32mL'image a été sauvegardé :{} detection : {}\033[0m".format(image_path,valid_format))
                                else :
                                    raise ValueError ("\033[31mFormat not valid {} =>ocr_result {} image {}  next image\033[0m".format(valid_format,ocr_result,image_path))
                                #crop an image based on coordinates
                                #cropobj = im0[int(xyxy[1])-10:int(xyxy[3])+10,int(xyxy[0])-10:int(xyxy[2])+10] 
                                #print(f"x : {xyxy[0]} y : {xyxy[1]}")
                            except Exception as e:
                                    print("\033[33mil y a une exception : {}  image : {} \033[0m".format(e,image_path))    
                                    try:
                                        os.remove(image_path)
                                    except OSError:
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
            #print(f' {s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

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

            """
            ### save image in the folder
            print("liste final des elements : ", Last_img_data)
            for data_image in Last_img_data:
                print("path",data_image[0], "image", data_image[-1])
                #cv2.imwrite(data_image[0],data_image[-1]) #save the crop .jpg file   cv2.imwrite(path, image coordinate)
            """
            ### save all data  in a csv (image name , detection result, GPS info, )
            csv_file_path =os.path.join(save_dir, "Detection_results.csv")
            with open(csv_file_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                writer.writerow(['image name', 'ocr detection'])
                for item in Last_img_data:
                    writer.writerow([os.path.basename(item[0]),item[1]])
            
            ### compare our database detection to the data of our research
            #csv1 with csv 2
            
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
import uuid
import time 
import re 
import pandas as pd
import difflib

dataf = {'name':['test'],'ocr':['PLkada']}
df_test = pd.DataFrame(dataf)
print(df_test)
dt_save = []
img_save = []
Last_img_data = []
#from threading import Thread
#from concurrent.futures import ThreadPoolExecutor
def reading_plate(image_path,save_dir):
    #ocr 
    print("launch OCR\n\n")
    #print(image_path)
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(image_path,paragraph="False", allowlist= "ABCDEFGHJKLMNPQRSTUWXYZ0123456789") # "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    #filtre palque FR formt     AA-000-AA sans o i et v 
    # if on u chiffre au lieu d'une lettre on remplace par une * pour dire 
    ocr_result = result[0][1]
    print("ocr result",ocr_result)

    

    #function compare
    if len(dt_save) != 0:  # check if list not empty
        temp_last = dt_save[-1]  # get last ocr text
        img_last = img_save[-1]

    dt_save.append(ocr_result)
    img_save.append(image_path)
    current_last = dt_save[-1]
    print('temp  last',temp_last , 'current last', current_last)
    print(difflib.SequenceMatcher(None, temp_last, current_last).ratio())
    compare_ratio = difflib.SequenceMatcher(None, temp_last, current_last).ratio()
    if compare_ratio ==1 or (len(dt_save)>=6 and compare_ratio <=0.7):  # if 2 strings same or if list size >= 6 and ratio <0.7 this mean change of license plate
        print("Diff = 1 ou 6 photos prises ")
        dt_save.clear()
        #save crop part
        crop_file_path = os.path.join(save_dir, str(time.strftime("%Y%m%d-%H%M%S"))+".jpg")
        Last_img_data.append([crop_file_path,img_last]) # save each photos with (path(Date) + ocr result +img save )

        #cv2.imwrite(crop_file_path,img_last) #save the crop .jpg file
        return Last_img_data
       # return temp_last 
    #print("ocr brut :",ocr_result)
    #text = filter_plate(ocr_result)
    #print("ocr filtre :",text)
    #return result
    #print(result)

        






def filter_plate(licence_text):
    print("filter")    
    if re.match("^[A-Z]{2}[0-9]{3}[A-Z]{2}$", licence_text):
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
        
    #^[A-Z]{2}-[0-9]{3}-[A-Z]{2}$

    # comparaison avec plaque précédente 
def save_data(ocr_text, img_name, csv_filename):
    #df = pd.DataFrame(columns=['Image','ocr detection'])
    print("save data")
    
    #new_row = {'name':os.path.basename(img_name), 'ocr':ocr_text}

    #df_test.append(new_row,ignore_index=True)
    #print(df_test)
    #print("end save data")
    #if os.path.exists(csv_filename):
    #    dat_to_csv = pd.read_csv(csv_filename, skipinitialspace=True, delim_whitespace=True)
     #   dat_to_csv.to_csv(csv_filename, index=None, columns=['image', 'OCR Result'])
     #   print(dat_to_csv)
    #with open(csv_filename, mode='r', errors="ignore") as file:
     #     final_line = file.readlines()[-1]
      #    print("oui",final_line)
    #print(final_line)
    print("oui ouvre pour ajouter une ligne") 
    
    #save data to csv  
    #with open(csv_filename, mode='a', newline='') as f:
    

   # with open(csv_filename, mode='a', newline='') as f:
    #    csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #    csv_writer.writerow([os.path.basename(img_name), ocr_text])
    # get last row to compare with
    

#######

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #####personal code addd
    # create csv file for save all data
    """print("initilisation csv file")
    csv_file_path = os.path.join(save_dir, "Detection_results.csv")
    #create csv
    with open(csv_file_path, mode='w') as f:
        dw = csv.DictWriter(f, delimiter=',',fieldnames=["photo","OCR TEXT"])
        dw.writeheader()"""
    
   


    ############
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

    #crp_cnt = 0
    crop_rate = 30   # every  frame that perfom analyse
    
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
                if not os.path.exists("crop"):
                        os.mkdir("crop")
                
                # Write results
                for *xyxy, conf, cls in reversed(det):

                    ############# personal modif
                    if opt.crop :
                        #print(f'{frame % crop_rate}')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                        if frame % crop_rate ==0 :
                            try:
                                print("on crop \n")
                                #crop an image based on coordinates
                                object_coordinates = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]
                                cropobj = im0[int(xyxy[1])-5:int(xyxy[3])+5,int(xyxy[0])-5:int(xyxy[2])+5]

                                

                                 #crop_file_path = os.path.join(save_dir, str(crp_cnt)+".jpg")
                                #crp_cnt = crp_cnt+1
                                csv_file_path = os.path.join(save_dir, "Detection_results.csv")
                                
                                #with open(csv_file_path, 'w') as f:
                                #    pass
                                try :
                                    #thread = Thread(target=reading_plate(crop_file_path), args=(i,))
                                   # thread.start()
                                   # ocr = Thread(target=reading_plate, args=(crop_file_path,))
                                    #ocr.start()
                                    """
                                    print (crop_file_path)
                                    List_img=[crop_file_path]
                                    with ThreadPoolExecutor(3) as exe:
                                        #exe.submit(reading_plate,2)
                                        result = exe.map(reading_plate,List_img)
                                    print("fin thread")
                                    for r in result:
                                        print(r)
                                    """
                                    ocr_detection = reading_plate(cropobj,save_dir)  #thread / video /database csv 
                                    print("ocr after",ocr_detection)
                                    #ocr_detection = reading_plate(crop_file_path)  #thread / video /database csv 
                                    """On compare la dectection avec la précédente en utilisant difflib 
                                    import difflib
                                    print(difflib.SequenceMatcher(None, 'PL123AK 75', 'JC858 P8050L').ratio())
                                    retourne un ratio de similitude si par exemple >0.5 on dit que bah beosin de d'nregistrer et donc on peut supprimer l'image
                                                                                                            """

                                   # data = save_data(ocr_detection[0][1], crop_file_path, csv_file_path)
                                   # print('data',data)
                                    
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
    parser.add_argument('--ocr', action='store_true', help='ocr text reading dectection')
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
    
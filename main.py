import config
import torch
import myutils
import easyocr
import pandas as pd
from PIL import Image
import time
import numpy as np
import sys
import pyautogui as pg
import cv2


if config.logmode:
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    log = open(config.dir_log+"log_"+timestr+".txt", "a")
    sys.stdout = log

########################################
# Step 0: 
#   OCR, YOLOv5, and other setups
########################################
reader = easyocr.Reader(['en'])

model_captchatype = torch.hub.load('./yolov5','custom',path='./weights/taskclassification.pt',source='local')
# For boats, bridges, crosswalks, chimneys, palm (trees), stairs
model_bbccps = torch.hub.load('./yolov5','custom',path='./weights/bbccps.pt',source='local')
# For others (See config.vovabulary)
model_others = torch.hub.load('./yolov5','custom',path='./weights/yolov5x.pt',source='local')

# Report model
if config.reportmodel:
    print(model_captchatype)
    print('==================================================')
    print('==================================================')
    print('==================================================')
    print(model_bbccps)
    print('==================================================')
    print('==================================================')
    print('==================================================')
    print(model_others)

# Currently, cannot run on my GPU (GPU memory issue)
model_captchatype.to(config.DEVICE)
model_bbccps.to(config.DEVICE)
model_others.to(config.DEVICE)

# Task round
task_round = 0

# Report
print("Step 0 finished.")
print("----------------------------------------")
print(" ")

########################################
# Step 1: 
#   Locate and click the check box, save the screenshot for inference
########################################
myutils.click_checkbox()
while True:
    task_round += 1

    # Random click mode will be activated if the task target is not supported.
    randomclickmode = False

    # Final click matrix init
    click_3x3 = myutils.click_matrix_init('3x3')
    click_4x4 = myutils.click_matrix_init('4x4')

    print("----------------------------------------")
    print(f"Task Round {task_round}")
    print("----------------------------------------")
    pg.screenshot(config.dir_img_sc_save+'screenshot_stage1.png')
    print("Screenshotted for task type inference.")

    # Report
    print(f"Round {task_round}: Step 1 finished.")
    print("----------------------------------------")
    print(" ")

    ########################################
    # Step 2: 
    #   Inference 1: 
    #       YOLOv5x: Tpye of recaptcha task
    #       OCR: Error info in recaptcha task window and recaptcha target
    ########################################
    # Loading screenshot from Step 1
    img = Image.open(config.dir_img_sc_save+'screenshot_stage1.png')

    # Test
    # img = Image.open('palm tree.png')
    # testresult = model_captchatype(img)
    # testresult = model(img)
    # testresult.save('./testresult/')
    # testresult.print()
    # print(testresult.pandas().xyxy[0])
    # print('Test end, exit.')
    # sys.exit()

    # Inference for recaptcha task and target area, get the area with highest confidence
    # result_inf_yolo_tasktype = model_captchatype(img).pandas().xyxy[0].sort_values(by=['confidence'],ascending=False).iloc[0,:]

    # Currently, cannot inference based on confidence, need more training data. But still can detect some areas, including the recaptcha area. So will calculate the areas and compare and then get the correct one.
    result_inf_yolo_classification = model_captchatype(img)
    result_inf_yolo_tasktype = myutils.inf_task_byarea(result_inf_yolo_classification.pandas().xyxy[0])

    if config.testmode_classification:
        print(result_inf_yolo_classification.pandas().xyxy[0])
        result_inf_yolo_classification.save(config.dir_testmode_classification)

    # Crop the screenshot for ocr inference and later 4x4 task (if it is 4x4)
    cord_crop_sc = result_inf_yolo_tasktype[0:4].apply(round)
    img_ocr = img.crop((cord_crop_sc))
    # Record abslute cord for later use (click)
    cord_origin_abs = cord_crop_sc[0:2]
    img_ocr.save(config.dir_img_sc_save+'crop_forocr.png') # Save for later check
    # OCR inference
    result_inf_ocr = reader.readtext(np.array(img_ocr), detail=0)
    if config.reportocr:
        print(result_inf_ocr)

    # Record inference results
    # 3x3 or 4x4
    task_type = result_inf_yolo_tasktype[-1]
    # Currently, small chance to predict 4x4 as 3x3. So make corrections manually. 
    # Later with more training data, this issue will be solved.
    if 'square' in result_inf_ocr[0] and task_type != '4x4':
        print("Task type correction...")
        task_type = '4x4'
    # 3x3 sub type: click until none left
    none_left_3x3 = True if 'left' in result_inf_ocr[2] else False

    # Get target object
    print(f"Target (OCR): {result_inf_ocr[1]}")
    task_info = myutils.vocabulary_change(result_inf_ocr[1].strip().replace("_",""))
    task_target = task_info[0]
    if task_info[1] == 'others':
        model = model_others
        print('Task target in pre-trained yolov5.')
    elif task_info[1] == 'bbccps':
        model = model_bbccps
        print('Task target in self-trained yolov5.')
    else:
        randomclickmode = True
        print('!!Task target currently not supported, click randomly!!')
    # Submit button status
    button_status = result_inf_ocr[-1]
    # Cord correction (if there's error info, relative cord will change)
    errinfo = True if 'Please' in str(result_inf_ocr).replace("'","").split() else False

    # Report
    print(f"Task type: {task_type}, tast target: {task_target}")
    print(f"Round {task_round}: Step 2 finished.")
    print("----------------------------------------")
    print(" ")

    ########################################
    # Step 3: 
    #   Get cord by type
    ########################################
    suborigin = myutils.locate_grid(task_type,errinfo)
    print(suborigin)
    print(f"Round {task_round}: Step 3 finished.")
    print("----------------------------------------")
    print(" ")

    ########################################
    # Step 4: 
    #   Inference by task type
    ########################################
    ########################################
    # Step 4a: 
    #   For 3x3 task: 
    #       Further crop
    #       Denoise
    #       Inference on 9 images
    #           a) 3x3 normal: click and go
    #           b) 3x3 non-left: click and again
    ########################################
    if task_type == "3x3" and not randomclickmode:
        print("A non left 3x3 task.") if none_left_3x3 else print("A normal 3x3 task.")
        width_subimg = 125
        imgs = []
        # Crop to 9 imgs, from upper left to lower right, 0-8
        for i in range(9):
            # img.crop((suborigin[i][0],suborigin[i][1],suborigin[i][0]+width_subimg,suborigin[i][1]+width_subimg)).save(config.dir_img_sc_crop+'crop_'+str(i)+'.png')
            imgs.append(img.crop((suborigin[i][0],suborigin[i][1],suborigin[i][0]+width_subimg,suborigin[i][1]+width_subimg)))

        # Denoise
        for i in range(9):
            imgs[i] = cv2.fastNlMeansDenoisingColored(np.array(imgs[i]),None,3,3)
            # Denoise Result test
            # testresult = model(imgstest)
            # testresult2 = model(imgs[0])
            # testresult.save('./result_3x3/')
            # testresult2.save('./result_3x3/')
            # imgs[0].save('ill.jpg')
            # imgs[0].show()
            # testresult.pandas().xyxy[0]
            # testresult2.pandas().xyxy[0]
        
        # Yolo inference on all 9 imgs
        print("Predicting...")
        time_inf_start = time.time()
        result_inf_yolo_3x3 = model(imgs)
        print("Inference time for 3x3 task: %s seconds." % (time.time() - time_inf_start))
        

        result_inf_yolo_3x3_df = pd.DataFrame()
        for i in range(9):
            if not result_inf_yolo_3x3.pandas().xyxy[i].empty:
                cache_df = result_inf_yolo_3x3.pandas().xyxy[i]
                cache_df['img'] = i
                result_inf_yolo_3x3_df = result_inf_yolo_3x3_df.append(cache_df)
        
        if config.testmode_3x3:
            # for i in range(9):
            #     print(result_inf_yolo_3x3.pandas().xyxy[i])
            #     print("======================================")

            print("======================================")
            print(result_inf_yolo_3x3_df)

            result_inf_yolo_3x3.save(config.dir_testmode_3x3)


        # Click inference
        result_inf_yolo_3x3_df_target = result_inf_yolo_3x3_df[result_inf_yolo_3x3_df['name']==task_target]
        print("Yolo detected target objects info:")
        print(result_inf_yolo_3x3_df_target)

        # Remove duplicates
        click_3x3_idx = list(dict.fromkeys(result_inf_yolo_3x3_df_target['img'].tolist()))

        # Confirmed clicks
        for i in range(len(click_3x3_idx)):
            click_3x3[click_3x3_idx[i]] = True
        
        print(click_3x3)

        # Normal 3x3
        if not none_left_3x3:
            # Normal 3x3 recaptcha need to submit at least 3 sub images,
            # so do click compensation if the normal 3x3 inference result 
            # contains 2 or less targets.
            if sum(click_3x3) < 3:
                print('Clicks < 3, so do random click compensation for 3x3...')
                click_3x3 = myutils.click_compensation_random(click_3x3)
                # Later other advanced improvements. For example, yolo may miss the motor when a person is on it. So this part is for corrections by task target.

            if errinfo:
                # If there's the error info, after click the first image, 
                # all of the cords will chagne. So need to click one first, then 
                # get new cords.
                print("Error info detected. After first click, adjust the cords.")
                newsuborigin = myutils.action_click_imgs_errinfo(task_type,click_3x3,suborigin)
                time.sleep(0.2)
                # Submit, but with new cords.
                myutils.action_click_submit(newsuborigin)

            else:
                # 3x3 final click action
                myutils.action_click_imgs(task_type,click_3x3,suborigin)
                
                # Click submit button
                myutils.action_click_submit(suborigin)

            # If success, exit
            myutils.check_success(suborigin)

            # # Check if there's any checked img
            # pg.screenshot(config.dir_img_sc_save+'screenshot_stage1.png')
            # print("Screenshotted for unclicking...")
            # checked_imgs = myutils.checked_imgs()
            # if checked_imgs:
            #     myutils.action_unclick_imgs(checked_imgs)
            # else:
            #     print("No checked sum imgs on screen. Going forward...")
        
        else:
            # None left 3x3:
            if any(click_3x3):
                # Still has something to click...
                myutils.action_click_imgs(task_type,click_3x3,suborigin)
                print("One round of non left 3x3 task finished.")
                # Since none left 3x3 has some animations, wait...
                time.sleep(0.5)
            else:
                # No more to click. Submit.
                print("No more to click in non left 3x3 task. Submit.")
                myutils.action_click_submit(suborigin)
                myutils.check_success(suborigin)


    ########################################
    # Step 4b: 
    #   For 4x4 task: 
    #       Inference
    #       Match with task target
    #       Locate boxes
    #       Click
    ########################################
    if task_type == "4x4" and not randomclickmode:
        width_subimg = 95

        # Inference for the big image
        #result_inf_yolo_4x4 = model('./cache_screenshot/motor.png') # For test
        time_inf_start = time.time()
        result_inf_yolo_4x4 = model(img_ocr)
        print("Inference time for 4x4 task: %s seconds." % (time.time() - time_inf_start))
        if config.testmode_4x4:
            result_inf_yolo_4x4.save(config.dir_testmode_4x4)

        result_inf_yolo_4x4_df = result_inf_yolo_4x4.pandas().xyxy[0]
        print(result_inf_yolo_4x4_df)

        # Get screen-level cord 
        result_inf_yolo_4x4_df['xmin'], result_inf_yolo_4x4_df['xmax'] = result_inf_yolo_4x4_df['xmin']+cord_origin_abs[0], result_inf_yolo_4x4_df['xmax']+cord_origin_abs[0]
        result_inf_yolo_4x4_df['ymin'], result_inf_yolo_4x4_df['ymax'] = result_inf_yolo_4x4_df['ymin']+cord_origin_abs[1], result_inf_yolo_4x4_df['ymax']+cord_origin_abs[1]
        print(result_inf_yolo_4x4_df)

        # Check if we have detected the target
        if task_target in result_inf_yolo_4x4_df['name'].tolist():
            # Get dataframe that only contains the target object
            result_inf_yolo_4x4_df_target = result_inf_yolo_4x4_df[result_inf_yolo_4x4_df['name'] == task_target]

            for obj_idx in range(result_inf_yolo_4x4_df_target.shape[0]):
                for subimg_idx in range(16):
                    # Intersection over image pieces
                    area = myutils.ioip(suborigin[subimg_idx],result_inf_yolo_4x4_df_target.iloc[obj_idx,:])
                    if area >= 0.05:
                        click_4x4[subimg_idx] = True
            print(click_4x4)

            if sum(click_4x4) < 2:
                print('Clicks < 2, so do random click compensation for 4x4...')
                click_4x4 = myutils.click_compensation_random(click_4x4,True)
                # Later other advanced improvements. For example, yolo may miss the motor when a person is on it. So this part is for corrections by task target.

            # 4x4 final click action
            if errinfo:
                # If there's the error info, Similar to 3x3 task previously...
                print("Error info detected. After first click, adjust the cords.")
                newsuborigin = myutils.action_click_imgs_errinfo(task_type,click_4x4,suborigin)
                time.sleep(0.2)
                # Submit, but with new cords.
                myutils.action_click_submit(newsuborigin)
            else:
                myutils.action_click_imgs(task_type,click_4x4,suborigin)
                # Click submit button
                myutils.action_click_submit(suborigin)

            # If success, exit
            myutils.check_success(suborigin)

            # # If reaches this step, means didn't pass the task.
            # # So, check if there's any checked img
            # pg.screenshot(config.dir_img_sc_save+'screenshot_stage1.png')
            # print("Screenshotted for unclicking...")
            # checked_imgs = myutils.checked_imgs()
            # if checked_imgs:
            #     myutils.action_unclick_imgs(checked_imgs)
            # else:
            #     print("No checked sum imgs on screen. Going forward...")            

        # If no target in the 4x4 image, skip
        else:
            print("No target detected in current 4x4 task. Skipping...")
            if errinfo:
                # If there's the error info, adjust the cords...
                print("Error info detected. adjusting the cords for skipping...")
                newsuborigin = myutils.locate_grid(task_type,errinfo)
                time.sleep(0.2)
                # Submit, but with new cords.
                myutils.action_click_submit(newsuborigin)   
            else:
                # Click submit button
                myutils.action_click_submit(suborigin)

            # Need to check success here since probably "no object in the 4x4 image" may be the ground truth.
            myutils.check_success(suborigin)

    ########################################
    # Step 4c: 
    #   Give up this round. Random click. 
    ########################################
    if randomclickmode:
        if task_type == "3x3":
            print("Random clicking 3x3 task...")
            click_3x3 = myutils.click_compensation_random(click_3x3)
            myutils.action_click_imgs(task_type,click_3x3,suborigin)
            time.sleep(1)
            print("Random clicking finished. Submit.")
            myutils.action_click_submit(suborigin)
            myutils.check_success(suborigin)
        if task_type =="4x4":
            print("Random clicking 4x4 task...")
            click_4x4 = myutils.click_compensation_random(click_4x4,True)
            myutils.action_click_imgs(task_type,click_4x4,suborigin)
            time.sleep(1)
            print("Random clicking finished. Submit.")
            myutils.action_click_submit(suborigin)
            myutils.check_success(suborigin)










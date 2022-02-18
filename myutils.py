import pyautogui as pg
import time
import sys
import config
import pandas as pd
import pyautogui
import cv2
import numpy as np
from random import randrange

######################################
# Check if checkbox is success
######################################
def check_success(suborigin):
    success = pg.locateOnScreen(config.dir_img_refs+'success.png')
    if success:
        print('recaptcha task success. No further tasks needed.')
        sys.exit()
    time.sleep(1)
    action_cleaning_errinfo(suborigin)
    time.sleep(1)

def temp_check_success():
    success = pg.locateOnScreen(config.dir_img_refs+'success.png')
    if success:
        print('recaptcha task success. No further tasks needed.')
        sys.exit()
    time.sleep(1)

######################################
# Find checked imgs
######################################
def checked_imgs(testmode = False):
    # img = cv2.imread('./test3.png') # For test
    img = cv2.imread(config.dir_img_sc_save+'screenshot_stage1.png')
    template = cv2.imread(config.dir_img_refs+'checked_imgs.png')
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.9)
    if testmode:
        w, h = template.shape[:-1]
        for pt in zip(*loc[::-1]):  # Switch collumns and rows
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            cv2.imwrite('./checkedimgresult.png',img)
    if loc[0].size == 0:
        return False
    else:
        imgloc = []
        for i in range(len(loc[0])):
            imgloc.append([loc[1][i],loc[0][i]]) # Notice: have to switch the cords
        return imgloc

######################################
# Click checkbox and take screenshot
######################################
def click_checkbox():
    print("Searching for checkbox...")
    # Locate a single FIXED object in a screenshot
    checkbox = pg.locateOnScreen(config.dir_img_refs+'checkbox_nm.png')
    if checkbox:
        print("New checkbox detected.")
        cordx,cordy = checkbox[0]+20,checkbox[1]+20
        pg.moveTo(cordx,cordy)
        print(f"Mouse move to {cordx},{cordy}")
        pg.click()
        print(f"Click {cordx},{cordy}")
        pg.moveTo(cordx-40,cordy-40)
        time.sleep(2) 
        temp_check_success()     

    else:
        checkbox = pg.locateOnScreen(config.dir_img_refs+'checkbox_waiting.png')
        if checkbox:
            print("Waiting checkbox detected.")
            cordx,cordy = checkbox[0]+20,checkbox[1]+20
            pg.moveTo(cordx,cordy)
            print(f"Mouse move to {cordx},{cordy}")
            pg.click()
            print(f"Click {cordx},{cordy}")
            pg.moveTo(cordx-40,cordy-40)
            time.sleep(2)
            temp_check_success()   
            
        else:
            checkbox = pg.locateOnScreen(config.dir_img_refs+'checkbox_expired.png')
            if checkbox:
                print("Expired checkbox detected.")
                cordx,cordy = checkbox[0]+20,checkbox[1]+20
                pg.moveTo(cordx,cordy)
                print(f"Mouse move to {cordx},{cordy}")
                pg.click()
                print(f"Click {cordx},{cordy}")
                pg.moveTo(cordx-40,cordy-40)
                time.sleep(2)
                temp_check_success()   

            else:
                print("No checkbox on screen found. Please check.")
                sys.exit()
            
########################################
# Locate the grid and return sub origins
########################################
def locate_grid(recaptchatype,errorinfo):
    '''
    recaptchatype: char, "3x3" or "4x4"
    errorinfo: bool
    '''
    print('Searching for refresh button...')
    # Locate a single FIXED object on screen
    refresh = pg.locateOnScreen(config.dir_img_refs+'refresh.png')
    if refresh:
        print('Refresh button detected.')
        # Get origin cord
        cordx,cordy = (refresh[0]-9, refresh[1]-441) if errorinfo else (refresh[0]-9, refresh[1]-411)
        suborigin_3x3 = [
            [cordx,cordy    ],[cordx+130,cordy    ],[cordx+260,cordy    ],
            [cordx,cordy+130],[cordx+130,cordy+130],[cordx+260,cordy+130],
            [cordx,cordy+260],[cordx+130,cordy+260],[cordx+260,cordy+260]
        ]
        suborigin_4x4 = [
            [cordx,cordy    ],[cordx+97,cordy    ],[cordx+194,cordy    ],[cordx+291,cordy    ],
            [cordx,cordy+97 ],[cordx+97,cordy+97 ],[cordx+194,cordy+97 ],[cordx+291,cordy+97 ],
            [cordx,cordy+194],[cordx+97,cordy+194],[cordx+194,cordy+194],[cordx+291,cordy+194],
            [cordx,cordy+291],[cordx+97,cordy+291],[cordx+194,cordy+291],[cordx+291,cordy+291]
        ]
        if recaptchatype == "3x3":
            # import time
            # for i in range(len(suborigin_3x3)):
            #     report_move(suborigin_3x3[i][0],suborigin_3x3[i][1])
            #     time.sleep(1)
            return suborigin_3x3
        else:
            # for i in range(len(suborigin_4x4)):
            #     report_move(suborigin_4x4[i][0],suborigin_4x4[i][1])
            #     time.sleep(1)
            return suborigin_4x4
    else:
        print("Cannot find refresh button on screen. Please check.")
        sys.exit()

########################################
# Task area inference
########################################
def inf_task_byarea(df):
    # Calculate the area difference between predicted area and true area
    df['score'] = (((df['xmax'] - df['xmin']).round())*((df['ymax'] - df['ymin']).round())-404*585).abs()
    return df.sort_values(by=['score'],ascending=True).iloc[0,:-1]
    
########################################
# Target object word correction
########################################
def vocabulary_change(target):
    # Need a list to execute "in"
    if target in config.vocabulary['pl'].tolist():
        # Return corrected word and which model is that type in
        return config.vocabulary[config.vocabulary['pl']==target].iloc[0,1:3]
    else:
        print('The target object is not in the list. Check the OCR result or list.')
        sys.exit()

########################################
# Intersection over img pieces
########################################
def ioip(cord_imgpiece,inf_result_slice):
    img_x1, img_x2 = cord_imgpiece[0], cord_imgpiece[0] + 95
    img_y1, img_y2 = cord_imgpiece[1], cord_imgpiece[1] + 95
    target_x1, target_y1 = inf_result_slice[0].round(), inf_result_slice[1].round()
    target_x2, target_y2 = inf_result_slice[2].round(), inf_result_slice[3].round()
    dx = min(img_x2,target_x2) - max(img_x1,target_x1)
    dy = min(img_y2,target_y2) - max(img_y1,target_y1)
    if dx > 0  and dy > 0:
        return dx * dy
    else:
        return 0

########################################
# Click action functions
########################################
def action_click_imgs(tasktype,status,suborigin):
    if tasktype == '4x4':
        for click in range(16):
            # Final 4x4 click
            if status[click]:
                pyautogui.moveTo(suborigin[click][0]+20,suborigin[click][1]+20)
                print(f"Mouse move to {suborigin[click][0]+20},{suborigin[click][1]+20}")
                time.sleep(0.2)
                pyautogui.click()
                print(f"Mouse click {suborigin[click][0]+20},{suborigin[click][1]+20}")
                time.sleep(0.2)
        print('4x4 click task finished.')
    
    elif tasktype == '3x3':
        for click in range(9):
            # Final 3x3 click
            if status[click]:
                pyautogui.moveTo(suborigin[click][0]+20,suborigin[click][1]+20)
                print(f"Mouse move to {suborigin[click][0]+20},{suborigin[click][1]+20}")
                time.sleep(0.2)
                pyautogui.click()
                print(f"Mouse click {suborigin[click][0]+20},{suborigin[click][1]+20}")
                time.sleep(0.2)
        print('3x3 click task finished.')

    else:
        print('Invalid task type. Please check.')
        sys.exit()

def action_click_imgs_errinfo(tasktype,status,suborigin_start):
    firstclick = np.where(status)[0][0]
    pyautogui.moveTo(suborigin_start[firstclick][0]+20, suborigin_start[firstclick][1]+20)
    print(f"Mouse move to {suborigin_start[firstclick][0]+20},{suborigin_start[firstclick][1]+20}")
    pyautogui.click()
    print(f"Mouse click {suborigin_start[firstclick][0]+20},{suborigin_start[firstclick][1]+20}")
    time.sleep(0.2)
    suborigin_new = locate_grid(tasktype,False)
    for i in range(1,sum(status)):
        currentclick = np.where(status)[0][i]
        pyautogui.moveTo(suborigin_new[currentclick][0]+20, suborigin_new[currentclick][1]+20)
        print(f"Mouse move to {suborigin_new[currentclick][0]+20},{suborigin_new[currentclick][1]+20}")
        pyautogui.click()
        time.sleep(0.2)
    return suborigin_new

def action_click_submit(suborigin):
        pyautogui.moveTo(suborigin[0][0]+325,suborigin[0][1]+425)
        print(f"Mouse move to {suborigin[0][0]+325},{suborigin[0][1]+425}")
        time.sleep(0.2)
        pyautogui.click()
        print(f"Mouse click {suborigin[0][0]+325},{suborigin[0][1]+425}")
        time.sleep(1)

def action_unclick_imgs(cords, errinfo = False,start=0):
    for i in range(start,len(cords)):
        pyautogui.moveTo(cords[i][0]+10,cords[i][1]+30)
        print(f"Mouse move to {cords[i][0]+10},{cords[i][1]+30}")
        time.sleep(0.2)
        pyautogui.click()
        print(f"Mouse click {cords[i][0]+10},{cords[i][1]+30}")

def action_cleaning_errinfo(suborigin):
    print("Cleaning click...")
    pyautogui.moveTo(suborigin[0][0]+40,suborigin[0][1]+40)
    print(f"Mouse move to {suborigin[0][0]+40},{suborigin[0][1]+40}")
    time.sleep(0.2)
    pyautogui.click()
    print(f"Mouse click {suborigin[0][0]+40},{suborigin[0][1]+40}")
    time.sleep(0.5)
    pyautogui.click()
    print(f"Mouse click {suborigin[0][0]+40},{suborigin[0][1]+40}")
    time.sleep(1)

########################################
# Click matrix init
########################################
def click_matrix_init(mattype):
    if mattype == '3x3':
        return [False,False,False,False,False,False,False,False,False]
    elif mattype == '4x4':
        return [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
    else:
        print('Check click_matrix_init() input.')
        sys.exit()

########################################
# Click matrix init
########################################
def click_compensation_random(clicks,four=False):
    if not four:
        if len(clicks) >= 3:
            while sum(clicks) < 3:
                flipped = randrange(len(np.where(~np.array(clicks))[0]))
                clicks[np.where(~np.array(clicks))[0][flipped]] = True
            return clicks
        else:
            print("Check the input of click_compensation_random()")
            sys.exit()
    else:
        if len(clicks) >= 2:
            while sum(clicks) < 2:
                flipped = randrange(len(np.where(~np.array(clicks))[0]))
                clicks[np.where(~np.array(clicks))[0][flipped]] = True
            return clicks
        else:
            print("Check the input of click_compensation_random()")
            sys.exit()


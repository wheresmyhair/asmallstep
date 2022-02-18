import pandas as pd
#################
# utils.py
#################
dir_img_refs = "./img_ref/"
dir_img_sc_save = "./cache_screenshot/"

#################
# main.py
#################
DEVICE = 'cpu'

dir_img_sc_crop = "./cache_screenshot/crops/"

list_exceptions = ['mountains or hills','chimney','stairs','bridge']

# If true, will save the inference result to the folders
testmode_classification = True
testmode_3x3 = True
testmode_4x4 = True
dir_testmode_3x3 = "./result_3x3/"
dir_testmode_4x4 = "./result_4x4/"
dir_testmode_classification = "./result_classification/"

# Log
logmode = False
dir_log = "./logs/"

# YOLO Model structure report
reportmodel = False

# OCR result print
reportocr = True

#################
# Vocabulary
#################
vocabulary = pd.DataFrame(
    [
    ('boats',               'boat',           'bbccps'),  # Later Model
    ('bridges',             'bridge',         'bbccps'),  
    ('chimneys',            'chimney',        'bbccps'),   
    ('crosswalks',          'crosswalk',      'bbccps'),  
    ('palm trees',          'palm',           'bbccps'),   
    ('trees',               'palm',           'bbccps'),   
    ('stairs',              'stairs',         'bbccps'),  
    ####################################################
    ('buses',               'bus',            'others'), 
    ('bus',                 'bus',            'others'), 
    ('taxis',               'car',            'others'), 
    ('cars',                'car',            'others'), 
    ('vehicles',            'car',            'others'), 
    ('bicycles',            'bicycle',        'others'), 
    ('traffic lights',      'traffic light',  'others'), 
    ('tractors',            'truck',          'others'), 
    ('a fire hydrant',      'fire hydrant',   'others'), 
    ('fire hydrants',       'fire hydrant',   'others'), 
    ('fire hydrant',        'fire hydrant',   'others'), 
    ('motorcycles',         'motorcycle',     'others'),
    ####################################################
    ('mountains or hills',  'mountain',       'giveup') 
    ],
    columns=['pl', 'sg', 'model'])
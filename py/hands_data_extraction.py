#!/usr/bin/python
#==============================================================#
#           EXTRACT DATA FROM IAMGES USING MEDIA PIPES         #                                        
#==============================================================#

#TODO <make method to generate diferent out file type>

from asyncore import write
from pickle import TRUE
import sys
import os
import glob

from matplotlib.pyplot import flag
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DIR_PATH = "./"
IS_SINGLE_FILE = False
OUTPUT_NAME = "out.txt"
DEFAULT_OUT_TYPE = "txt"
SHOW_PROCESSED_IMAGE = False
SHOW_CONSOLE_LOG = False

if(os.path.exists("labels.txt")): os.remove("labels.txt")
output_labels_file = open("labels.txt", 'a')


def get_landmarks(dir_name, output_file, flags, label):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        #IMAGE_FILES = list(filter(lambda file_name: '.png' in file_name or '.jpg' in file_name, os.listdir(dir_name)))
        IMAGE_FILES = list(filter(lambda file_name: '.png' in file_name or '.jpg' in file_name, os.listdir(dir_name)))
        #print(list(filter(lambda file_name: '.png' in file_name or '.jpg' in file_name, IMAGE_FILES)))

        for idx, file in enumerate(IMAGE_FILES):

            image = cv2.flip(cv2.imread(dir_name+'/'+file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            #print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                #print("     ERROR: Not possible to classify image -> ", file)
                continue
            
            #print("     SUCCESS: processed image -> ", file)
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                if(flags['show_console_log']):
                    print('hand_landmarks:', hand_landmarks)
                    print(
                        f'Index finger tip coordinates: (',
                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    )
                save_landmarks(hand_landmarks, output_file, label)
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            #IF WE WANT TO SEE PROCESSED IAMGE OUTPUT
            if(not flags['show_processed_image']): continue

            cv2.imwrite(
            '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
            # Draw hand world landmarks.
            if not results.multi_hand_world_landmarks:
                continue
            for hand_world_landmarks in results.multi_hand_world_landmarks:
                mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

    
def get_flags(flags):
    #default flag values
    flags_obj = {
        "dir_path": "./",
        "is_single_file" : IS_SINGLE_FILE,
        "output_name_file" : OUTPUT_NAME,
        "output_type" : DEFAULT_OUT_TYPE, #MAKE ENUM HERE
        "show_processed_image" : SHOW_PROCESSED_IMAGE,
        "show_console_log" : SHOW_CONSOLE_LOG,
    }

    if('-s' in flags): flags_obj['is_single_file'] = True
    if('-d' in flags): flags_obj['dir_path'] = flags[flags.index("-d")+1]
    if('-o' in flags): flags_obj['output_name_file'] = flags[flags.index("-o")+1]
    if('-i' in flags): flags_obj['show_processed_image'] = True
    if('-c' in flags): flags_obj['show_console_log'] = True
    
    return flags_obj

def save_landmarks(landmarks, output_file, label):
    landmarks_array = []
    for landmark in landmarks.landmark:
        new_array = [landmark.x, landmark.y, landmark.z]
        landmarks_array.append(new_array)
    output_file.write(str(landmarks_array))
    output_file.write(",\n")

    output_labels_file.write(label)
    output_labels_file.write(",\n")


def read_all_images(path, output_file, flags):
    #POR DEFAULT OS CAMINHO TEM QUE SEMPRE INCLUIR /
    label = path.split('/')[-1]
    images = list(filter(lambda f: os.path.isfile(os.path.join(path+'/', f)) ,os.listdir(path+'/')))
    if(len(images) == 0):
        print("(I) NO IMAGES IN: {}".format(path))
    else:
        print("(I) label ({}) -> READING IMAGES IN {}".format(label, path))
        get_landmarks(path, output_file, flags, label)

        

    folders = list(filter(lambda f: os.path.isdir(os.path.join(path+'/', f)) ,os.listdir(path+'/')))
    if len(folders) > 0: 
        print("(F) READING FOLDERS IN {}: {}".format(path, folders))
        for folder in folders:
            read_all_images(os.path.join(path+'/', folder), output_file, flags)

        

def read_one_image(path, show_processed_image):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        image = cv2.flip(cv2.imread(path), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            print("ERROR: Not possible to classify image -> ", path)
            return

        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_array = []
            for landmark in hand_landmarks.landmark:
                new_array = [landmark.x, landmark.y, landmark.z]
                landmarks_array.append(new_array)


        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        print("Hand Data: ", landmarks_array)

        if(not show_processed_image): return landmarks_array

        cv2.imwrite(
        '/tmp/annotated_image' + str(path.split("/")[-1]) + '.png', cv2.flip(annotated_image, 1))
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            return landmarks_array
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(
            hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

        return landmarks_array

    
def remove_last_character(out_file_name, out_labels_file_name):
    with open(out_file_name, 'rb+') as out_file:
        out_file.seek(-2, os.SEEK_END)
        out_file.truncate()

    with open(out_labels_file_name, 'rb+') as out_file:
        out_file.seek(-2, os.SEEK_END)
        out_file.truncate()

def main():
    flags = get_flags(sys.argv)
    #print("Provided Flags: "+str(flags))
    if(os.path.exists(flags['output_name_file'])): os.remove(flags['output_name_file'])
    output_file = open(flags['output_name_file'], 'a')


    if flags['is_single_file']: read_one_image(flags["dir_path"], flags['show_processed_image'])
    else: read_all_images(flags["dir_path"], output_file, flags)

    output_file.close()
    output_labels_file.close()

    

main()

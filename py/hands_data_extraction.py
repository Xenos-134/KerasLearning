#!/usr/bin/python

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


def get_landmarks(dir_name, output_file, flags):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        IMAGE_FILES = list(filter(lambda file_name: '.png' in file_name or '.jpg' in file_name, os.listdir(dir_name)))
        #list(filter(lambda file_name: '.png' in file_name or '.jpg' in file_name, os.listdir(dir_name)))
        print(list(filter(lambda file_name: '.png' in file_name or '.jpg' in file_name, IMAGE_FILES)))
        
        for idx, file in enumerate(IMAGE_FILES):

            image = cv2.flip(cv2.imread(dir_name+'/'+file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                print("ERROR: Not possible to classify image -> ", file)
                continue

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
                save_landmarks(hand_landmarks, output_file)
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

    if('-s' in flags): flags_obj['is_single_file'] = IS_SINGLE_FILE
    if('-d' in flags): flags_obj['dir_path'] = flags[flags.index("-d")+1]
    if('-o' in flags): flags_obj['output_name_file'] = flags[flags.index("-o")+1]
    if('-i' in flags): flags_obj['show_processed_image'] = True
    if('-c' in flags): flags_obj['show_console_log'] = True
    
    return flags_obj

def save_landmarks(landmarks, output_file):
    for landmark in landmarks.landmark:
        print("LANDMARK: ", landmark)
        output_file.write("{},".format(str(landmark.x)))
        output_file.write("{},".format(str(landmark.y)))
        output_file.write(str(landmark.z))
    output_file.write("\n")


def read_all_images(path):
    #POR DEFAULT OS CAMINHO TEM QUE SEMPRE INCLUIR /
    label = path.split('/')[-1]
    images = list(filter(lambda f: os.path.isfile(os.path.join(path+'/', f)) ,os.listdir(path+'/')))
    if(len(images) == 0):
        print("(I) NO IMAGES IN: {}".format(path))
    else:
        print("(I) label ({}) -> READING IMAGES IN {}: {}".format(label, path, images))
        

    folders = list(filter(lambda f: os.path.isdir(os.path.join(path+'/', f)) ,os.listdir(path+'/')))
    print("(F) READING FOLDERS IN {}: {}".format(path, folders))

    for folder in folders:
        read_all_images(os.path.join(path+'/', folder))





def train(input_file) :
    print("==================( CNN Training (START) )==================")
    #TODO
    print("==================( CNN Training   (END) )==================")
    


def main():
    flags = get_flags(sys.argv)
    print("Provided Flags: "+str(flags))
    if(os.path.exists(flags['output_name_file'])): os.remove(flags['output_name_file'])
    output_file = open(flags['output_name_file'], 'a')

    read_all_images(flags["dir_path"])
    #get_landmarks(flags["dir_path"], output_file, flags)



main()

import subprocess

#def install(name):
 #   subprocess.call(['pip', 'install', name])
    
#install ("cvlib")
#install ("opencv-python==4.5.5.64")
#install("streamlit")
#install ("dictdiffer")
#install("smtplib")
# read input image
import cv2
import argparse
import numpy as np
from google.colab.patches import cv2_imshow
from PIL import Image, ImageOps
import streamlit as st
import dictdiffer
#import smtplib

def vehicle_detection(uploaded_file):
    
    #import smtplib
    #st.write("def works")
    if uploaded_file is not None:
        #st.write("Took input")
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        input_image=image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input**\n")
            st.image(input_image)
        
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        
        # read class names from text file
        classes = None
        with open('custom_vehicle.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # generate different colors for different classes 
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        
        # read pre-trained model and config file
        net = cv2.dnn.readNet('custom_vehicle.cfg', 'custom_vehicle.weights')
        
        # create input blob 
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        
        # set input blob for the network
        net.setInput(blob)
        # function to get the output layer names 
        # in the architecture
        def get_output_layers(net):
            
            layer_names = net.getLayerNames()
            
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
            return output_layers
        
        # function to draw bounding box on the detected object with class name
        def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        
            label = str(classes[class_id])
        
            color = COLORS[class_id]
        
            cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        
            cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))
        
        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.3
        nms_threshold = 0.4
        
        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        # apply non-max suppression
        
        print("class_id",class_ids)
        
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        new_class_ids1=[]
        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            #i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            new_class_ids1.append(class_ids[i])
            draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            
        print("new class_id",new_class_ids1)
        #st.image(image, caption='Output', use_column_width=True)
        output_image=image
        #st.image([input_image,output_image])
        
        with col2:
            st.markdown("**Output**\n")
            st.image(output_image)
            return(new_class_ids1)
            
def comparision(result1, result2):
    map_dict = {0: 'car',
        1: 'truck',
        2: 'pickup',
        3:  'tractor',
        4: 'camping car',
        5: 'boat',
        6: 'other'
                  } 

    class_list1=[]
    dict1={}
    dict2={}
    class_list2=[]
    
    for image1 in result1:
        class_list1.append(map_dict[image1])
        #print("class_id1",class_ids)
        #print("class_list1",class_list1)
        
    for image2 in result2:
        class_list2.append(map_dict[image2])
        #print("class_id2",class_ids1)
        #print("class_list2",class_list2)
        
    for key1 in class_list1:
          
          if key1 not in dict1:
              dict1[key1] = 1
          else:
              dict1[key1] += 1
    
    for key2 in class_list2:
      
          if key2 not in dict2:
              dict2[key2] = 1
          else:
              dict2[key2] += 1 
          
          
    #st.write(dict1)
    #st.write(dict2)
    
    diff_list=[]
    for diff in list(dictdiffer.diff(dict1, dict2)):         
          diff_list.append(diff)
          
    #st.write(diff_list)
        
    st.write("Compared to the previous image: ")
    if len(diff_list)==0:
        st.write("There are no differences")
    else:
        for item in diff_list:
            for detection in item[2]:
                if item[0]=='add':
                    if detection[1]>1:
                        suffix='s'
                        prefix='are'
                    else:
                        suffix=''
                        prefix ='is'
      
                    st.write(f"There {prefix} {str(detection[1])} more {str(detection[0])}{suffix}")

                if item[0]=='remove':
                    if detection[1]>1:
                        suffix='s'
                        prefix='are'
                    else:
                        suffix=''
                        prefix ='is'
                    st.write(f"There {prefix} {str(detection[1])} less {str(detection[0])}{suffix}")

            if item[0] == 'change':
                if abs(item[2][1]-item[2][0])>1:
                    suffix='s'
                    prefix='are'
                else:
                    suffix=''
                    prefix ='is'

                if item[2][1]-item[2][0]>1:
                    st.write(f"There {prefix} {abs(item[2][1]-item[2][0])} more {item[1]}{suffix}")
                else:
                    st.write(f"There {prefix} {abs(item[2][1]-item[2][0])} less {item[1]}{suffix}")

        

    
    
    

        
        
                







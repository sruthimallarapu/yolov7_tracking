#Install and Import the Required Modules
import subprocess
from matplotlib import pyplot
import rasterio
def install(name):
    subprocess.call(['pip', 'install', name])
    
install ("cvlib")
install ("opencv-python==4.5.5.64")
install("streamlit")
install ("dictdiffer")
#install("smtplib")
# read input image
import cv2
import argparse
import numpy as np
from google.colab.patches import cv2_imshow
from PIL import Image, ImageOps
import streamlit as st
import dictdiffer
import tempfile
import sys
import os
from vehicle_functions import *
from building import *

#import streamlit as st
#from PIL import Image
#import tempfile
#import subprocess
#import sys
#import os



st.sidebar.title('Smart Farm Surveillance')
st.sidebar.subheader('Settings')
#genre = st.sidebar.selectbox("Choose your preference",('About Smart Farm Surveillance','Livestock Classification & Counting with behavior monitoring','Livestock Behavior Monitoring','Building Detection in Farm'))
base_genre = st.sidebar.selectbox("Choose your preference",('About Smart Farm Surveillance','Livestock Monitoring','Object Detection in Farm'))

if base_genre == 'Livestock Monitoring':

	with st.container():
		st.markdown("<h1 style='text-align: center; color: black;'>Livestock Monitoring</h1>", unsafe_allow_html=True)
		st.markdown(''' Here the type of cattle is identified using the drone video of cattle grazing on the farmland. Cattles like cow, sheep and horse are identified.
			            The basic positional behavior of cattles like standing, eating and idle states are  captured.
			            Along with the mentioned classification, the cattles are counted and their movement is also tracked.
			            The output classification video display either all the cattle types or the cattle type given by the user.
                    ''')
		genre = st.sidebar.radio('Choose the livestock classification type',('Species Classification','Behavior Classification'))

		if genre == 'Species Classification':
			st.markdown("<h3 style='text-align: center; color: black;'>Species Classification</h3>", unsafe_allow_html=True)
			#class_list_id = []
			#class_list = []	
			names = ['All','cow','sheep','horse']
			assigned_class = st.sidebar.radio("Choose the cattle to be identified:", names)
			#for each in assigned_class:
			#	class_list_id.append(names.index(each))
			#class_list_id.sort()

			#for index in class_list_id:
			#	class_list.append(names[index])
		
			st.sidebar.markdown('---')
			uploaded_files = st.sidebar.file_uploader("Upload the drone video of farm", type=[ "mp4", "mov",'avi','asf', 'm4v' ], accept_multiple_files=False)

			tfflie = tempfile.NamedTemporaryFile(delete=False)

			if not uploaded_files:
				pass
			else:
				tfflie.write(uploaded_files.read())
    
				dem_vid = open(tfflie.name, 'rb')
				demo_bytes = dem_vid.read()

				st.sidebar.text("Input Video")
				st.sidebar.video(demo_bytes)
				st.write("Processing video...")
				if assigned_class == 'All':
					subprocess.run(["python", "object_tracker.py","--mode","species","--model","yolov4","--video",tfflie.name, "--output", "species_results.avi","--weights","./checkpoints_species/yolov4-416"])
				else:
					subprocess.run(["python", "object_tracker.py","--mode","species","--model","yolov4","--video",tfflie.name, "--classlist",assigned_class,"--output", "species_results.avi","--weights","./checkpoints_species/yolov4-416"])
				st.success('Video is Processed')
		

			if (os.path.isfile("species_results.avi")):
				subprocess.run(["ffmpeg", "-y","-loglevel","panic","-i","species_results.avi", "species_results.mp4"])
				video_file = open("species_results.mp4", "rb") #enter the filename with filepath
				video_bytes = video_file.read()
				st.video(video_bytes)
			else:
				pass

		else:

			st.markdown("<h3 style='text-align: center; color: black;'>Behavior Classification</h3>", unsafe_allow_html=True)
			#class_list_id = []
			#class_list = []	
			names = ['All','Stand','Eat','Idle']
			assigned_class = st.sidebar.radio("Choose the behavior type to be identified:", names)
			#for each in assigned_class:
			#	class_list_id.append(names.index(each))
			#class_list_id.sort()

			#for index in class_list_id:
			#	class_list.append(names[index])
		
			st.sidebar.markdown('---')
			uploaded_files = st.sidebar.file_uploader("Upload the drone video of farm", type=[ "mp4", "mov",'avi','asf', 'm4v' ], accept_multiple_files=False)

			tfflie = tempfile.NamedTemporaryFile(delete=False)

			if not uploaded_files:
				pass
			else:
				tfflie.write(uploaded_files.read())
    
				dem_vid = open(tfflie.name, 'rb')
				demo_bytes = dem_vid.read()

				st.sidebar.text("Input Video")
				st.sidebar.video(demo_bytes)
				st.write("Processing video...")
				if assigned_class == 'All':
					subprocess.run(["python", "object_tracker.py","--mode","behavior","--model","yolov4","--video",tfflie.name, "--output", "behavior_results.avi","--weights","./checkpoints_behavior/yolov4-416"])
				else:
					subprocess.run(["python", "object_tracker.py","--mode","behavior","--model","yolov4","--video",tfflie.name, "--classlist",assigned_class,"--output", "behavior_results.avi","--weights","./checkpoints_behavior/yolov4-416"])
				st.success('Video is Processed')
		

			if (os.path.isfile("behavior_results.avi")):
				subprocess.run(["ffmpeg", "-y","-loglevel","panic","-i","behavior_results.avi", "behavior_results.mp4"])
				video_file = open("behavior_results.mp4", "rb") #enter the filename with filepath
				video_bytes = video_file.read()
				st.video(video_bytes)
			else:
				pass

    

elif base_genre == 'Object Detection in Farm':

	with st.container():
		st.markdown("<h1 style='text-align: center; color: black;'>Object Detection in Farm</h1>", unsafe_allow_html=True)
		st.markdown(''' Here the buildings and vehicles are identified from the satellite images.
						The satellite images are also compared with previous day's images to show any differences in the location of vehicles in the farm.
                    ''')

		genre = st.sidebar.radio('Choose the Object type for Detection',('Building','Vehicle'))
		
		if genre == 'Building':

			st.markdown("<h3 style='text-align: center; color: black;'>Building Detection</h3>", unsafe_allow_html=True)
			uploaded_files = st.sidebar.file_uploader("Upload a satellite image to detect buildings", type=[ "jpg", "tiff",'png' ], accept_multiple_files=False)
			tfflie = tempfile.NamedTemporaryFile(delete=False)
			if not uploaded_files:
				pass
			else:
				col1, col2 = st.columns(2)
				tfflie.write(uploaded_files.read())
				img = rasterio.open(tfflie.name)
				array = img.read(1)
				g = pyplot.imshow(array, cmap='pink')
				fig = g.figure
				with col1:
				    st.markdown('**Input**\n')
				    st.pyplot(fig)
				#inp_image = Image.open('building_image.jpg')
				#st.sidebar.image(inp_image)
				subprocess.run(["cp",tfflie.name,"input_image.tif"])
				load_model()
				if (os.path.isfile("building_prediction.png")):
				    with col2:
				       st.markdown('**Output**\n') 
				       st.image("building_prediction.png",width=500,use_column_width=True)
				
				

		else:
		    st.markdown("<h1 style='text-align: center; color: black;'>Vehicle Intrusion Detection and Classification</h1>", unsafe_allow_html=True)
		    #st.text("Upload a top view satellite image, png")
		    #uploaded_file = st.sidebar.file_uploader("Upload a satellite image to detect buildings", type=[ "jpg", "jpeg",'png' ]      ,accept_multiple_files=False)
		    #uploaded_file2 = st.sidebar.file_uploader("Upload a second satellite image to detect buildings and compare the results", type=[ "jpg", "jpeg",'png' ],accept_multiple_files=False)
		    uploaded_file = st.sidebar.file_uploader("Upload a top view satellite image", type="png")
		    uploaded_file2 = st.sidebar.file_uploader("Upload satellite image for comparison", type="png")
		    if uploaded_file is not None:
		        result1=vehicle_detection(uploaded_file)
		    if uploaded_file2 is not None:
		        result2=vehicle_detection(uploaded_file2)
		        comparision(result1, result2)
  
else:

	with st.container():
		st.markdown("<h1 style='text-align: center; color: black;'>Smart Farm Surveillance</h1>", unsafe_allow_html=True)
		image = Image.open('Search-for-Lost-Cattle.jpeg')
		st.image(image)
		st.markdown('**Smart Farm Surveillance** system is an alternative to the traditional approach of farm monitoring by the use of technology-oriented agricultural techniques.\n')
		st.markdown('The main of this surveillance system is to reduce the manual effort required for security of the farm. \n')
		st.markdown('The current version of this Smart Farm Surveillance system performs the tasks of classifying the cattle & its behavior along with object detection by the use of deep learning models.\n')


	
	
	

    
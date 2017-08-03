import cv2
from PIL import Image
import os
import numpy as np
from emotions import EMOTIONS

def get_image_data(filename):
	frame=cv2.imread(filename,0)
	return frame
def extract_filename(file):
	if(file.count("/")==0 or file.rindex("/")==len(file)-1):
		return None
	else:
		return file[file.rindex("/")+1:]
def gif_2_jpg(file,dest_dir):
	img=Image.open(file)
	bg=Image.new("RGB",img.size,(50,50,50))
	bg.paste(img,img)
	filename=extract_filename(file)
	if(filename.count(".")==0):
		filename+=".jpg"
	elif(filename.rindex(".")==len(filename)-1):
		filename+="jpg"
	else:
		rindex=filename.rindex(".")
		filename=filename[:rindex]+".jpg"
	bg.save(dest_dir+"/"+filename)
def group_pictures(source_folder,dest_folder):
	for f in os.listdir(source_folder):
		if(not os.path.isdir(source_folder+"/"+f)):
			lindex=f.index(".")
			rindex=f.rindex(".")

			label=f[lindex+1:rindex]

			if(not os.path.exists(dest_folder+"/"+label)):
				os.mkdir(dest_folder+"/"+label)
			os.rename(source_folder+"/"+f,dest_folder+"/"+label+"/"+f)
def save_data(frame,label_name,output_file):
	with open(output_file,"a") as f:
		frame=cv2.resize(frame,(32,24))
		frame=np.reshape(frame,(1,frame.shape[0]*frame.shape[1]))
		label=[0,0,0,0,0,0]
		label[EMOTIONS[label]]=1

		label=np.array(label).reshape(6,1)
		frame=np.concatenate((frame,label),axis=1)
		np.savetxt(f,frame,"%s",",")


def process_images(images_folder,output_file):
	for imfolder in os.listdir(images_folder):
		if EMOTIONS.has_key(imfolder):
			for imfile in os.listdir(images_folder+"/"+imfolder):
				frame=get_image_data(images_folder+"/"+imfolder+"/"+imfile)
				save_data(frame,imfolder,output_file)
def tiff_2_jpg(file,dest_dir):
	if(file.count(".tiff")==0):
		return
	im=Image.open(file)
	im.thumbnail(im.size)


	filename=extract_filename(file)
	if(filename.count(".")==0):
		filename+=".jpg"
	elif(filename.rindex(".")==len(filename)-1):
		filename+="jpg"
	else:
		rindex=filename.rindex(".")
		filename=filename[:rindex]+".jpg"
	img=im.convert("RGB")
	print(dest_dir + "/" + filename)
	img.save(dest_dir + "/" + filename, "JPEG", quality=100)


# def load_data():
# 	pass

for f in os.listdir("jaffeimages/jaffe"):
	if(not os.path.isdir("jaffeimages/jaffe/"+f)):
		if(not os.path.isdir("jaffeimages/jaffe/"+f)):
			tiff_2_jpg("jaffeimages/jaffe/"+f,"jaffeimages/jaffe/jpg")

# get_image_data("train/jpg/subject01.happy.jpg")

# group_pictures("train/jpg","train/jpg/grouped")

# frames=[]
# for i in os.listdir("train/jpg/grouped/happy"):
# 	frame=get_image_data("train/jpg/grouped/happy/"+i)
# 	frame=cv2.resize(frame,(32,24))
# 	frames.append(frame)
# 	# cv2.imshow("frame",frame)
# 	# cv2.waitKey(0)
# 	# cv2.destroyAllWindows()
# 	print(frame.shape)

# save_data(frames,"data.csv")

# process_images("train/jpg/grouped","data.csv")
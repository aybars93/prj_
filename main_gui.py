# importing tkinter and tkinter.ttk 
# and all their functions and classes 
from tkinter import * 
from tkinter.ttk import *

# importing askopenfile function 
# from class filedialog 
from tkinter.filedialog import askopenfile 
from tkinter.filedialog import askdirectory

import model_train

root = Tk() 
# root.geometry('200x100') 
root.geometry('500x300') 

# This function will be used to open 
# file in read mode and only Python files 
# will be opened 

path_1 = ""
path_2 = ""
def get_director_path(): 
	global path_1
	global path_2
	# file = askopenfile(mode ='r', filetypes =[('Python Files', '*.py')]) 
	dirname = askdirectory()
	print("Directory has been selected .. \n")	
	if(path_1 == ""):
		path_1 = dirname
	else:
		path_2 = dirname
	
	"""For file reading """
	# if file is not None: 
		# content = file.read() 
		# print(content) 

def start_training(): 
	global path_1
	global path_2
	print("Training has been started.. \n"),
	path_observation_1 = "D:\\Datasets\\archive_kaggle\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_005\\BraTS20_Training_005_t1.nii_folder\\"
	path_observation_2 = "D:\\Datasets\\archive_kaggle\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_005\\BraTS20_Training_005_t2.nii_folder\\"
	
	# path_observation_1 = ""
	# path_observation_2 = ""
	# model_train.train_model(path_observation_1, path_observation_2, epochs=3000, plot=False)
	print("\n path_1: selected as : " + str(path_1))
	print("\n path_2: selected as : " + str(path_2))
	model_train.train_model(path_1+"//", path_2+"//", epochs=3000, plot=False)
	# if(path_1 != ""):
		# myLabel_1 = Label(root,text="Path for Observation 1:\n"+str(path_1))
		# myLabel_1.pack(side = TOP, pady = 10)	
	# if(path_2 != ""):
		# myLabel_2=Label(root,text="Path for Observation 2:\n"+str(path_2))
		# myLabel_2.pack(side = TOP, pady = 10)		



btn = Button(root, text ='Select Observation 1 Folder', command = lambda:get_director_path()) 
btn.pack(side = TOP, pady = 10) 
btn = Button(root, text ='Select Observation 2 Folder', command = lambda:get_director_path()) 
btn.pack(side = TOP, pady = 10) 

btn = Button(root, text ='Start Training', command = lambda:start_training()) 
btn.pack(side = TOP, pady = 10) 
# file_dir_name = "Path : " + str(file_dir_name)
# if(file_dir_name != ""):
	# myLabel=Label(root,text=file_dir_name)
	# myLabel.pack()

mainloop() 






































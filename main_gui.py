"""
	Main module for GUI interaction to data selection and training for model
"""
from tkinter import * 
from tkinter.ttk import *
# from tkinter.filedialog import askopenfile 		
from tkinter.filedialog import askdirectory

import model_train

root = Tk() 
root.geometry('200x200') 
# root.geometry('500x300') 


path_1 = ""
path_2 = ""
def get_director_path(): 
	global path_1
	global path_2
	dirname = askdirectory()
	print("Directory has been selected .. \n")	
	if(path_1 == ""):
		path_1 = dirname
	else:
		path_2 = dirname
	


def start_training(): 
	global path_1
	global path_2
	print("\n path_1: selected as : " + str(path_1))
	print("\n path_2: selected as : " + str(path_2))
	print("Training has been started.. \n")
	
	model_train.train_model(path_1+"//", path_2+"//", 64, 3, epochs=10000, plot=False)



btn_1 = Button(root, text ='Select Observation 1 Folder', command = lambda:get_director_path()) 
btn_2 = Button(root, text ='Select Observation 2 Folder', command = lambda:get_director_path()) 
btn_3 = Button(root, text ='Start Training', command = lambda:start_training()) 
btn_1.pack(side = TOP, pady = 10) 
btn_2.pack(side = TOP, pady = 10) 
btn_3.pack(side = TOP, pady = 10) 


mainloop() 






































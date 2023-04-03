import cv2
import os.path
import pandas as pd

from tqdm import tqdm
from functions import getLocalSymmetry

FOLDER_PATH = 'symmetry_database'
COLUMNS = ['fileName','numSymmetries','numDecoys','symmetries','Darkness','xPeriod','yPeriod','turbSize','offsetX','offsetY','backgroundType']
SHAPE = (224,224,3)
MNIST = pd.read_csv('MNIST/mnist_test.csv')
MNIST.index.name = 'index'

# Check if folder to store the data exist
if not os.path.isdir(FOLDER_PATH):
    os.mkdir(FOLDER_PATH)
# Check if folder for images exist
if not os.path.isdir(os.path.join(FOLDER_PATH, 'images')):
    os.mkdir(os.path.join(FOLDER_PATH, 'images'))
# Check if folder for labels exist
if not os.path.isdir(os.path.join(FOLDER_PATH, 'labels')):
    os.mkdir(os.path.join(FOLDER_PATH, 'labels'))
# Check if labels exist
if not os.path.isfile(os.path.join(FOLDER_PATH, 'labels.csv')):
    temp = pd.DataFrame(columns=COLUMNS)
    temp.index.name = 'index'
    temp.to_csv(os.path.join(FOLDER_PATH, 'labels.csv'))

# Reading previous data
prevData = pd.read_csv(os.path.join(FOLDER_PATH,'labels.csv'),index_col='index')

# Checking starting point for loop
start = len(prevData)

# Loop
for i in tqdm(range(start,len(MNIST))):
    # File name
    fileName = f'{i}.png'
    # Generating local symmetry
    img, dictSym, dictBack = getLocalSymmetry(SHAPE, MNIST, idx=i)
    # Removing and modifying unnecesary data
    for dict in dictSym:
        dict['overFlow'] = int(dict['overFlow'] == True)
    # Adding data
    dictBack['fileName'] = fileName
    # Adding all simetries
    dictBack['numSymmetries'] = len(dictSym)
    dictBack['symmetries'] = str(dictSym)
    # Appending to df and saving
    newRowDf = pd.DataFrame(dictBack, index=[0])
    
    prevData = pd.concat([prevData, newRowDf], ignore_index=True)
    prevData.index.name = 'index'
    prevData.to_csv(os.path.join(FOLDER_PATH, 'labels.csv'))
    # Saving image
    cv2.imwrite(os.path.join('symmetry_database','images',fileName), img)  
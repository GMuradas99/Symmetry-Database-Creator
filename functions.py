### IMPORTS ###
import numpy as np
import cv2

### GETTERS ###

#Returns numpy array of the digit and its label
def getImageArray(id, minst):
    image = minst.iloc[id][1:].values.flatten().tolist()
    array = np.reshape(image, (int(len(image)**0.5),int(len(image)**0.5)))

    return cv2.merge((array,array,array)).astype(np.uint8), minst.iloc[id]['label']

# Returns the starting and ending points for the symmetry axis as well as the coordinates for the bounding box
def getSAandBB(img):
    minX = len(img[0])*10
    minY = -1
    maxX = -1
    maxY = -1
    for i in range(len(img)):
        for j in range(len(img[0])):
            if minY == -1 and (img[i][j][0] != 0 or img[i][j][1] != 0 or img[i][j][2] != 0):
                minY = i
            if img[i][j][0] != 0 or img[i][j][1] != 0 or img[i][j][2] != 0:
                maxY = i-1
                if j-1 > maxX:
                    maxX = j
                if j < minX:
                    minX = j

    # Axis of symmetry
    startAxis = (minX + (maxX-minX)//2, minY)            
    endAxis = (minX + (maxX-minX)//2, maxY)
    
    return startAxis,endAxis,minX,maxX,minY,maxY

### OPERATIONS ###

#Returns the combination of the two images with no overflow
def addNoOverflow(img1, img2):
    result = np.zeros(img1.shape, dtype=np.uint8)

    for i in range(len(img1)):
        for j in range(len(img1[0])):
            for x in range(len(img1[0][0])):
                val = int(img1[i][j][x]) + int(img2[i][j][x])
                if val > 255:
                    val = 255
                result[i][j][x] = val

    return result

#Returns the combination of img1 and img2 with the selected padding
def addWithPadding(img1,img2,padding,overFlow=False):
    # Create a black pixel column with the same height and color channels as the original image
    black_pixels = np.zeros((img1.shape[0], abs(padding), img1.shape[2])).astype(np.uint8)

    # Concatenate the black pixel column to the left and right of the images
    if padding>=0:
        padded_image1 = np.concatenate((img1, black_pixels), axis=1)
        padded_image2 = np.concatenate((black_pixels,img2), axis=1)
    else:
        padded_image1 = np.concatenate((black_pixels, img1), axis=1)
        padded_image2 = np.concatenate((img2, black_pixels), axis=1)

    #Returns the selected concatenation
    if not overFlow:
        return addNoOverflow(padded_image1,padded_image2)
    else:
        return np.add(padded_image1,padded_image2)

# Returns the rotated image
def rotateDigit(img, degrees):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, degrees, 1)

    return cv2.warpAffine(img, rotation_matrix, (width, height)), rotation_matrix

# Adds padding to both axis of the image in order to safely perform rotation
def addRotationPadding(img):
    # Calculating the necessary padding
    height, width = img.shape[:2]
    diagonal = int((height**2+width**2)**0.5)+2
    addWidth = (diagonal-width)//2+1
    addHeight = (diagonal-height)//2+1

    # Applying padding along the X axis
    black_pixels_horizontal = np.zeros((img.shape[0], addWidth, img.shape[2])).astype(np.uint8)

    padded_image = np.concatenate((img, black_pixels_horizontal), axis=1)
    padded_image = np.concatenate((black_pixels_horizontal, padded_image), axis=1)

    # Applying padding along the Y axis
    black_pixels_vertical = np.zeros((addHeight, padded_image.shape[1], img.shape[2])).astype(np.uint8)

    padded_image = np.concatenate((padded_image, black_pixels_vertical), axis=0)
    padded_image = np.concatenate((black_pixels_vertical, padded_image), axis=0)

    return padded_image

# Returns a list with all the transformmed keypoints
def transformKeypoints(keypoints, rotationMatrix):
    result = []
    for keypoint in keypoints:
        rotatedPoint = rotationMatrix.dot(np.array(keypoint + (1,)))
        result.append((rotatedPoint[0],rotatedPoint[1]))

    return result
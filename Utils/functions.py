### IMPORTS ###

import cv2
import ast
import math
import rpack
import random
import numpy as np

from os.path import join

### GETTERS ###

#Returns numpy array of the digit and its label
def getImageArray(id, mnist):
    image = mnist.iloc[id][1:].values.flatten().tolist()
    array = np.reshape(image, (int(len(image)**0.5),int(len(image)**0.5)))

    return cv2.merge((array,array,array)).astype(np.uint8), mnist.iloc[id]['label']

# Returns random image of the desired number
def getRandomMNISTDigit(mnist, label):
    labeled = mnist.loc[mnist['label'] == label]
    image = labeled.sample().values.flatten().tolist()[1:]
    array = np.reshape(image, (int(len(image)**0.5),int(len(image)**0.5)))

    return cv2.merge((array,array,array)).astype(np.uint8)

# Returns the starting and ending points for the symmetry axis as well as the center, width and height for the bounding box
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
    maxX += 1
    maxY += 1

    # Axis of symmetry
    startAxis = (minX + (maxX-minX)/2, minY)            
    endAxis = (minX + (maxX-minX)/2, maxY)

    # Center, width and height of bounding box
    center = ((maxX-minX)/2+minX, (maxY-minY)/2+minY)
    height = (maxY-minY)
    width = (maxX-minX)
    
    return startAxis,endAxis,center,width,height

# Returns the starting and ending points for the symmetry axis in cross symmetries as well as the coordinates for the bounding box
def getCrossSAandBB(img):
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
    maxX += 1
    maxY += 1

    # Axis of symmetry
    startAxis1 = (minX + (maxX-minX)//2, minY)            
    endAxis1 = (minX + (maxX-minX)//2, maxY)

    startAxis2 = (minX, minY + (maxY-minY)//2)            
    endAxis2 = (maxX, minY + (maxY-minY)//2)
    
    # Center, width and height of bounding box
    center = ((maxX-minX)/2+minX, (maxY-minY)/2+minY)
    height = (maxY-minY)
    width = (maxX-minX)
    
    return [[startAxis1, endAxis1],[startAxis2, endAxis2]],center,width,height

# Returns a one dimensional gradient
def get_gradient_line(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T

# Returns a two dimensional gradient
def get_gradient(width, height, start_list = None, stop_list = None, is_horizontal_list = None):
    if start_list is None:
        start_list = (random.randrange(255), random.randrange(255), random.randrange(255))
    if stop_list is None:
        stop_list = (random.randrange(255), random.randrange(255), random.randrange(255))
    if is_horizontal_list is None:
        is_horizontal_list = (random.getrandbits(1), random.getrandbits(1), random.getrandbits(1))
    result = np.zeros((height, width, len(start_list)), dtype=np.uint8)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_line(start, stop, width, height, is_horizontal)

    return result

# Returns bounding box points given parameters
def getBoundingBoxPoints(center: list, width: int, height: int, rotation: float):
    # Points for bounding box
    pts = [(center[0]-width/2 , center[1]-height/2), (center[0]+width/2 , center[1]-height/2), 
           (center[0]+width/2 , center[1]+height/2), (center[0]-width/2 , center[1]+height/2)]
    
    # Rotating
    rotationMatrix = cv2.getRotationMatrix2D(center, rotation, 1)
    return transformKeypoints(pts, rotationMatrix)

# Draws bounding box and symmetry axis from dictionary
def drawFromDict(src: np.ndarray, dict: dict):
    drawPolygon(src, getBoundingBoxPoints(dict['center'], dict['width'], dict['height'], dict['finalRotation']))
    drawLines(src, dict['symAxes'])

### DISPLAY FUNCTIONS

# Draws all lines in list on specified color
def drawLines(img: np.ndarray, lines: list, color: list[int] = [255,0, 0]):
    for [startAxis, endAxis] in lines:
        cv2.line(img, (int(startAxis[0]) , int(startAxis[1])), (int(endAxis[0]) , int(endAxis[1])), color, 1)
    return img

# Draws polygon on the image
def drawPolygon(img: np.ndarray, points: list, color: list[int] = [0, 255, 0]):
    for i in range(len(points)):
        if i != len(points)-1:
            cv2.line(img, (int(points[i][0]),int(points[i][1])), (int(points[i+1][0]),int(points[i+1][1])), color, 1)
        else:    
            cv2.line(img, (int(points[i][0]),int(points[i][1])), (int(points[0][0]),int(points[0][1])), color, 1)

# Draws symmetry axis and bounding box
def drawSAandBB(img, startAxis, endAxis, center, width, height, rotation):
    # Symmetry axis
    drawLines(img, [[startAxis, endAxis]])

    # Getting points
    pts = getBoundingBoxPoints(center, width, height, rotation)

    # Drawing bounding box
    drawPolygon(img, pts)

# Displays all bounding boxes and symmetry axises
def drawAllSAandBB(img, symDictionaries):
    for dic in symDictionaries:
        drawMultipleSAandBB(img, dic['symAxes'], dic['center'], dic['width'], dic['height'], dic['finalRotation'])
    return img

# Draws multiple symmetry axis and bounding box
def drawMultipleSAandBB(img, axes, center, width, height, rotation):
    # Symmetry axis
    drawLines(img, axes)

    # Getting points
    pts = getBoundingBoxPoints(center, width, height, rotation)

    # Drawing bounding box
    drawPolygon(img, pts)

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

# Returns a list with all the transformed keypoints
def transformKeypoints(keypoints, rotationMatrix):
    result = []
    for keypoint in keypoints:
        if type(keypoint) == tuple:
            rotatedPoint = rotationMatrix.dot(np.array(keypoint + (1,)))
        elif type(keypoint) == list:
            rotatedPoint = rotationMatrix.dot(np.array(keypoint + [1,]))
        elif type(keypoint) == np.ndarray:
            rotatedPoint = rotationMatrix.dot(np.append(keypoint, [1.]))
        else: 
            assert('Not supported data type')
        result.append((rotatedPoint[0],rotatedPoint[1]))
        
    return result

# Remove the excess padding from image
def removePadding(img, startAxis, endAxis, cent):
    _,_, center, width, height = getSAandBB(img)
    minX = int(center[0] - width/2)
    minY = int(center[1] - height/2)
    maxX = int(center[0] + width/2)
    maxY = int(center[1] + height/2)
    cropped = img[minY:maxY, minX:maxX]
    newStartX = startAxis[0] - minX
    newStartY = startAxis[1] - minY
    newEndX = endAxis[0] - minX
    newEndY = endAxis[1] - minY
    newCentX = cent[0] - minX 
    newCentY = cent[1] - minY

    return cropped, (newStartX,newStartY), (newEndX,newEndY), (newCentX, newCentY)

# Remove the excess padding from image
def removePaddingMultipleAxes(img, symAxes, cent):
    _,_, center, width, height = getSAandBB(img)
    minX = int(center[0] - width/2)
    minY = int(center[1] - height/2)
    maxX = int(center[0] + width/2)
    maxY = int(center[1] + height/2)
    cropped = img[minY:maxY, minX:maxX]
    newSymAxes = []
    for [startAxis, endAxis] in symAxes:
        newStartX = startAxis[0] - minX
        newStartY = startAxis[1] - minY
        newEndX = endAxis[0] - minX
        newEndY = endAxis[1] - minY
        newSymAxes.append([[newStartX, newStartY],[newEndX, newEndY]])
    newCentX = cent[0] - minX 
    newCentY = cent[1] - minY

    return cropped, newSymAxes, (newCentX, newCentY)

#Resizes the image and keypoints according to the selected percent
def resizeSymmetry(percent, img, startAxis, endAxis, center, inWidth, inHeight):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)

    newStartX = startAxis[0] * percent / 100
    newStartY = startAxis[1] * percent / 100
    newEndX = endAxis[0] * percent / 100
    newEndY = endAxis[1] * percent / 100
    newCenterX = center[0] * percent / 100
    newCenterY = center[1] * percent / 100
    newWidth = inWidth * percent / 100
    newHeight = inHeight * percent / 100

    return cv2.resize(img, (width, height)), (newStartX,newStartY), (newEndX,newEndY), (newCenterX, newCenterY), newWidth, newHeight

#Resizes the image and keypoints according to the selected percent
def resizeSymmetryMultipleAxes(percent, img, symAxes, center, inWidth, inHeight):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)

    newSymAxes = []
    for [startAxis, endAxis] in symAxes:
        newStartX = startAxis[0] * percent / 100
        newStartY = startAxis[1] * percent / 100
        newEndX = endAxis[0] * percent / 100
        newEndY = endAxis[1] * percent / 100
        newSymAxes.append([[newStartX, newStartY],[newEndX, newEndY]])
    newCenterX = center[0] * percent / 100
    newCenterY = center[1] * percent / 100
    newWidth = inWidth * percent / 100
    newHeight = inHeight * percent / 100

    return cv2.resize(img, (width, height)), newSymAxes, (newCenterX, newCenterY), newWidth, newHeight

# Adds noise filter to image
def addNoise(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            pixelMask = random.uniform(0,1)
            img[i][j][0] += int(255 * pixelMask)
            img[i][j][1] += int(255 * pixelMask)
            img[i][j][2] += int(255 * pixelMask)
    
    return img

# Generates smooth noise
def smoothNoise(x, y, img):
    width = img.shape[1]
    height = img.shape[0]

    # Get decimal part for x and y
    fractX = x - int(x)
    fractY = y - int(y)

    # Wrap around
    x1 = (int(x) + width) % width
    y1 = (int(y) + height) % height

    # Neighbour values
    x2 = (x1 + width - 1) % width
    y2 = (y1 + height -1) % height

    # Smooth noise
    value = 0.0
    value += fractX * fractY * img[y1][x1][0]
    value += (1 - fractX) * fractY * img[y1][x2][0]
    value += fractX * (1 - fractY) * img[y2][x1][0]
    value += (1 - fractX) * (1 - fractY) * img[y2][x2][0]

    return value

# Smooth the image to desired factor
def smoothImage(img,factor):
    result = np.zeros(img.shape).astype(np.uint8)

    # Applies smooth noise
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res = int(smoothNoise(i/ factor, j / factor, img))
            result[i][j][0] = res
            result[i][j][1] = res
            result[i][j][2] = res

    return result

# Returns the value for the turbulence pixel
def turbulence(x, y, size, img):
    value = 0
    initialSize = size

    while size >= 1:
        value += (smoothNoise(x / size, y / size, img) / 256) * size
        size = size//2
    
    return  128 * value / initialSize

# Applies turbulence
def applyTurbulence(img, initialSize):
    turb = np.zeros(img.shape).astype(np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res = int(turbulence(i, j, initialSize, img))
            turb[i][j][0] = res
            turb[i][j][1] = res
            turb[i][j][2] = res

    return turb

# Returns the sin texture
def sinTexture(shape, factorX, factorY, power):
    img = np.zeros(shape).astype(np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = int(255 * abs(math.sin(math.radians(i*factorX+j*factorY))) ** power)
            img[i,j][0] = val
            img[i,j][1] = val
            img[i,j][2] = val
    
    return img

# Returns the wood texture
def woodTexture(shape, factorX, factorY, offsetX, offsetY, power):
    img = np.zeros(shape).astype(np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = int(255 * abs(math.cos(math.radians(    math.sqrt(((i+offsetX)*factorX)**2+((j+offsetY)*factorY)**2)     ))) ** power)
            img[i,j][0] = val
            img[i,j][1] = val
            img[i,j][2] = val
    
    return img

# Inserts into image by replacing the pixels with the desiredposition
def insert(insert, img, position):
    img[position[0]:position[0]+insert.shape[0], position[1]:position[1]+insert.shape[1]] = insert

# Returns image of selected size with randomly placed digits and a symmetries as well as the data for the symmetries
def getRandomDigitsWithSymmetry(id, mnist, size, types = ['simple', 'cross'], weights = [0.5, 0.5], numSymmetries = None, initialRotation = None, 
                                overFlow = None, padding1 = None, padding2 = None, finalRotation = None, resizingPercent = None):
    # Elements for while loop
    possible = True
    squareSizes = []
    elements = []
    # symmetries = []
    symDictionaries = []

    if numSymmetries is None:
        numSymmetries = random.randrange(1,5)

    # Adding symmetries and placing them in the first place of the list
    symmetry,symDictionary = createAnySymmetry(id, mnist, types, weights, initialRotation, overFlow, padding1, padding2, finalRotation, resizingPercent)
    squareSizes.append(symmetry.shape[:2])
    elements.append(symmetry)
    symDictionaries.append(symDictionary)
    for _ in range(numSymmetries - 1):
        symmetry,symDictionary = createAnySymmetry(random.randrange(len(mnist)), mnist, types, weights, initialRotation, overFlow, padding1, padding2, finalRotation, resizingPercent)
        squareSizes.append(symmetry.shape[:2])
        elements.append(symmetry)
        symDictionaries.append(symDictionary)

    positions = []
    # Attemting to add elements until impossible
    while possible:
        square = createAsymmetry(mnist)
        try: 
            positions = rpack.pack(squareSizes+[square.shape[:2]],max_width=size[1],max_height=size[0])
            squareSizes.append(square.shape[:2])
            elements.append(square)
        except rpack.PackingImpossibleError:
            possible = False

    # Final result
    background = np.zeros((size[1],size[0],3)).astype(np.uint8)

    # Inserting the digits on their specific position
    for idx,digit in enumerate(elements):
        insert(digit,background,positions[idx])

    for i in range(numSymmetries):
        adjusted = []
        for [startAxis, endAxis] in symDictionaries[i]['symAxes']:
            adjusted.append([
                [startAxis[0] + positions[i][1], startAxis[1] + positions[i][0]],
                [endAxis[0] + positions[i][1], endAxis[1] + positions[i][0]]
            ])
        symDictionaries[i]['symAxes'] = adjusted
        symDictionaries[i]['center'] = (symDictionaries[i]['center'][0] + positions[i][1], symDictionaries[i]['center'][1] + positions[i][0])
    
    return background, symDictionaries, len(positions)-len(symDictionaries)

# Draws all symmetries in the row on the img
def drawRow(img, row):
    # Transforming list of dictionaries drom str to list
    symmetries = ast.literal_eval(row['symmetries'])
    # Looping through all symmetries in the image
    for symm in symmetries:
        drawMultipleSAandBB(img, symm['symAxes'], symm['center'], symm['width'], symm['height'], symm['finalRotation'])
    return img

# Returns a mask of the selected row
def getMask(row, path, thickness = 2):
    img = cv2.imread(join(path, row['fileName']))
    mask = np.zeros((img.shape[:2]), np.uint8)
    # Painting axis on mask
    symmetries = ast.literal_eval(row['symmetries'])
    # Looping through all symmetries in the image
    for symm in symmetries:
        for [startAxis,endAxis] in symm['symAxes']:
            cv2.line(mask, (int(startAxis[0]),int(startAxis[1])), (int(endAxis[0]),int(endAxis[1])), 255, thickness)
    return mask

# Applies random color gradient to given image
def applyColorGradient(image, start=None, end=None, axes=None):
    grad = get_gradient(image.shape[1],image.shape[0], start_list=start, stop_list=end, is_horizontal_list=axes)
    norm = grad/255
    image = np.multiply(image,norm)
    image = image.astype(np.uint8)

    return image

# Applies a repliclable random color gradient to given image, input the return tuple to replicate the gradient in another object
def applyReplicableColorGradient(image, replicate = None):
    if replicate is None:
        replicate = {
            'start' : (random.randrange(255), random.randrange(255), random.randrange(255)),
            'end'   : (random.randrange(255), random.randrange(255), random.randrange(255)),
            'axes'  : (random.getrandbits(1), random.getrandbits(1), random.getrandbits(1))
        }
    image = applyColorGradient(image, start=replicate['start'], end=replicate['end'], axes=replicate['axes'])

    return image, replicate

# Adds insert into image so both coordinates coincide
def insertPointOnPoint(ins, pointInsert, image, pointImage):
    insertCoord = (pointImage[0]-pointInsert[0], pointImage[1]-pointInsert[1])
    insert(ins, image, insertCoord)

# Performs set step in rotation
def performRotationStep(img, rotAxis, order, step):
    rotationStep = 360.0 / order
    rotationMatrix = cv2.getRotationMatrix2D(rotAxis, rotationStep * step, 1)
    return cv2.warpAffine(img, rotationMatrix, img.shape[:2])

# Adds all images in list on top of each other
def addAllImages(listImages, overflow = True):
    result = listImages[0].copy()
    for img in listImages[1:]:
        if overflow:
            result = np.add(result, img)
        else:
            result = addNoOverflow(result,img)
    return result

# Returns minimum height and width of a group of images
def getMinMaxHeightAndWidth(images: list) -> tuple:
    minY = images[0].shape[0]
    minX = images[0].shape[1]
    maxY = 0
    maxX = 0
    for image in images:
        if minY > image.shape[0]:
            minY = image.shape[0]
        if minX > image.shape[1]:
            minX = image.shape[1]
        if maxY < image.shape[0]:
            maxY = image.shape[0]
        if maxX < image.shape[1]:
            maxX = image.shape[1]
    return (minY, minX), (maxY, maxY)

# Skews image to desired specifications, retunrs skew token for display
def skewImage(src: np.ndarray, width: int, height: int, center, rotation: float, skewPercX: int, skewPercY: int, axis: str= 'Vertical') -> np.ndarray:
    # Gathering input box
    input_pts = getBoundingBoxPoints(center, width, height, 0)

    # Skewing box
    if axis == 'Vertical':
        output_pts = np.float32([
            [(input_pts[0][0] + (width/2)*skewPercX), (input_pts[0][1] + (height/2)*skewPercY)],
            input_pts[1],
            [(input_pts[2][0] - (width/2)*skewPercX), (input_pts[2][1] - (height/2)*skewPercY)],
            input_pts[3]
        ])
    else:
        output_pts = np.float32([
            input_pts[0],
            [(input_pts[1][0] + (width/2)*skewPercX), (input_pts[1][1] + (height/2)*skewPercY)],
            input_pts[2],
            [(input_pts[3][0] + (width/2)*skewPercX), (input_pts[3][1] + (height/2)*skewPercY)],
        ])
    
    # Rotating box
    rotationMatrix = cv2.getRotationMatrix2D(center, rotation, 1)
    rotated_input =  np.float32(transformKeypoints(input_pts, rotationMatrix))
    output_pts =  np.float32(transformKeypoints(output_pts, rotationMatrix))

    # Applying transform
    M = cv2.getPerspectiveTransform(rotated_input, output_pts)
    out = cv2.warpPerspective(src=src,M=M,dsize=(src.shape[1], src.shape[0]),flags=cv2.INTER_LINEAR)

    # Skew Token
    skewToken = {
        'axis': axis,
        'skewPercX': skewPercX,
        'skewPercY': skewPercY,
    }

    return out, skewToken

# Takes a list of segments and returns a list of points
def fromSegmentsToPoints(segments: list):
    points = []
    for symAxis in segments:
        points.append(symAxis[0])
        points.append(symAxis[1])
    return points

# Takes a list of points and returns a list of segments
def fromPointsTosegments(points: list):
    segments = []
    for i in range(len(points)//2):
        segments.append([points[i*2],points[i*2+1]])
    return segments

# Skews the point to the specified token
def skewPoints(pts: list, width: int, height: int, center, rotation: float, skewToken: dict):
    input_pts = getBoundingBoxPoints(center, width, height, 0)

    # Skewing box
    if skewToken['axis'] == 'Vertical':
        output_pts = np.float32([
            [(input_pts[0][0] + (width/2)*skewToken['skewPercX']), (input_pts[0][1] + (height/2)*skewToken['skewPercY'])],
            input_pts[1],
            [(input_pts[2][0] - (width/2)*skewToken['skewPercX']), (input_pts[2][1] - (height/2)*skewToken['skewPercY'])],
            input_pts[3]
        ])
    else:
        output_pts = np.float32([
            input_pts[0],
            [(input_pts[1][0] + (width/2)*skewToken['skewPercX']), (input_pts[1][1] + (height/2)*skewToken['skewPercY'])],
            input_pts[2],
            [(input_pts[3][0] + (width/2)*skewToken['skewPercX']), (input_pts[3][1] + (height/2)*skewToken['skewPercY'])],
        ])
    
    # Rotating box
    rotationMatrix = cv2.getRotationMatrix2D(center, rotation, 1)
    rotated_input =  np.float32(transformKeypoints(input_pts, rotationMatrix))
    output_pts =  np.float32(transformKeypoints(output_pts, rotationMatrix))

    # Creating transform Matrix
    M = cv2.getPerspectiveTransform(rotated_input, output_pts)

    return transformKeypoints(pts, M)

### MAIN FUNCTIONS ###

# Creates a symmetry with the selected weight bias
def createAnySymmetry(id, mnist, types , weights, initialRotation = None, overFlow = None, padding1 = None, padding2 = None, 
                      finalRotation = None, resizingPercent = None, color = True):
    choice = random.choices(types, weights=weights, k=1)[0]

    if choice == 'simple':
        symmetry,symDictionary = createSymmetry(id,mnist,initialRotation,overFlow,padding1,finalRotation,resizingPercent,color)
        symAxes = [[[symDictionary['startAxis'][0], symDictionary['startAxis'][1]] , [symDictionary['endAxis'][0], symDictionary['endAxis'][1]]]]
        del symDictionary['startAxis']
        del symDictionary['endAxis']
        symDictionary['symAxes'] = symAxes
    if choice == 'cross':
        symmetry, symDictionary = createCrossSymmetry(id, mnist, initialRotation, overFlow, padding1, padding2, finalRotation, resizingPercent, color)

    return symmetry, symDictionary

# Creates a random symmetry, returns array with image, its symmetry axis and its label; parameters can be modified.
def createSymmetry(id, minst, initialRotation = None, overFlow = None, padding = None, finalRotation = None, resizingPercent = None, color = True):
    # Getting the image and label
    result,label = getImageArray(id,minst)
    if color:
        result = applyColorGradient(result)
    
    # Initial rotation
    if initialRotation is None:
        initialRotation = random.randrange(360)
    result,_ = rotateDigit(result, initialRotation)

    # Mirroring the image
    mirrored = cv2.flip(result, 1)

    # Combining initial with mirrored 
    if overFlow is None:
        overFlow = bool(random.getrandbits(1))
    if padding is None:
        padding = random.randrange(-result.shape[0], result.shape[0])
    result = addWithPadding(result,mirrored, padding, overFlow=overFlow)

    # Adding padding for rotation
    result = addRotationPadding(result)

    # Obtaining symmetry axis
    startAxis, endAxis, center, width, height = getSAandBB(result)

    # Final rotation
    if finalRotation is None:
        finalRotation = random.randrange(360)
    result,rotationMatrix = rotateDigit(result,finalRotation)

    # Rotating symmetry axis
    rotated = transformKeypoints([startAxis, endAxis, center], rotationMatrix)
    startAxis = rotated[0]
    endAxis = rotated[1]
    center = rotated[2]

    # Remove excess pading
    result, startAxis, endAxis, center = removePadding(result, startAxis, endAxis, center)

    # Resizing
    if resizingPercent is None:
        resizingPercent = random.randrange(80,300)
    result, startAxis, endAxis, center, width, height = resizeSymmetry(resizingPercent, result, startAxis, endAxis, center, width, height)

    return result, {'startAxis':startAxis, 'endAxis': endAxis, 'center':center, 'width':width, 'height':height, 'label':label, 'initialRotation':initialRotation, 'overFlow':overFlow, 'padding':padding, 'finalRotation':finalRotation, 'resizingPercent':resizingPercent}

# Creates cross symmetry
def createCrossSymmetry(id,minst,initialRotation = None, overFlow = None, padding1 = None, padding2 = None, finalRotation = None, 
                        resizingPercent = None, color = True):
    result,label = getImageArray(id,minst)
    if color:
        result = applyColorGradient(result)

    # Initial rotation
    if initialRotation is None:
        initialRotation = random.randrange(360)
    result,_ = rotateDigit(result, initialRotation)

    # Vertically mirroring the image
    mirrored = cv2.flip(result, 1)

    # Combining initial with mirrored 
    if overFlow is None:
        overFlow = bool(random.getrandbits(1))
    if padding1 is None:
        padding1 = random.randrange(-result.shape[0], result.shape[0])
    result = addWithPadding(result,mirrored, padding1, overFlow=overFlow)

    # Horizontally mirroring the image
    mirrored = cv2.flip(result, 0)

    # Rotating both images 90º so they can be added with function
    mirrored = cv2.rotate(mirrored, cv2.ROTATE_90_CLOCKWISE)
    result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

    # Combining initial with mirrored
    if padding2 is None:
        padding2 = random.randrange(-result.shape[0], result.shape[0])
    result = addWithPadding(result,mirrored, padding1, overFlow=overFlow)
    
    # Adding padding for rotation
    result = addRotationPadding(result)

    # Obtaining symmetry axes
    symAxes, center, width, height = getCrossSAandBB(result)

    # Final rotation
    if finalRotation is None:
        finalRotation = random.randrange(360)
    result,rotationMatrix = rotateDigit(result,finalRotation)

    # Rotating symmetry axis
    rotated = transformKeypoints([symAxes[0][0], symAxes[0][1], symAxes[1][0], symAxes[1][1], center], rotationMatrix)
    symAxes[0][0] = rotated[0]
    symAxes[0][1] = rotated[1]
    symAxes[1][0] = rotated[2]
    symAxes[1][1] = rotated[3]
    center = rotated[4]

    # Remove excess pading
    result, symAxes, center = removePaddingMultipleAxes(result, symAxes, center)

    # Resizing
    if resizingPercent is None:
        resizingPercent = random.randrange(80,300)
    result, symAxes, center, width, height = resizeSymmetryMultipleAxes(resizingPercent, result, symAxes, center, width, height)

    return result, {'symAxes':symAxes, 'center':center, 'width':width, 'height':height, 'label':label, 'initialRotation':initialRotation, 'overFlow':overFlow,
                    'padding1':padding1, 'padding2': padding2, 'finalRotation':finalRotation, 'resizingPercent':resizingPercent}

# Returns random smooth sin texture
def getSmoothNoiseSin(shape, darkness = None, xPeriod = None, yPeriod = None, turbPower = None, turbSize = None):
    if darkness is None:
        darkness = random.uniform(0,0.8)
    if xPeriod is None:
        xPeriod	= random.randrange(10)
    if yPeriod is None:
        yPeriod	= random.randrange(10)
    if turbPower is None:
        turbPower = random.uniform(0,3)
    if turbSize is None:
        turbSize = 2**random.randrange(2,7)
    
    noise = np.zeros(shape).astype(np.uint8)
    noise = addNoise(noise)

    img = np.zeros((224,224,3)).astype(np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            xyValue = i * xPeriod + j * yPeriod + (turbPower * turbulence(i,j, turbSize, noise))
            sineValue = 255 * abs(math.sin(math.radians(xyValue)))**2 * darkness
            img[i,j][0] = sineValue
            img[i,j][1] = sineValue
            img[i,j][2] = sineValue

    return img, {'Darkness':darkness, 'xPeriod':xPeriod, 'yPeriod':yPeriod, 'turbPower':turbPower, 'turbSize':turbSize}

# Returns random smooth sin texture
def getSmoothNoiseWood(shape, offsetX = None, offsetY = None, darkness = None, xPeriod = None, yPeriod = None, turbPower = None, turbSize = None):
    if darkness is None:
        darkness = random.uniform(0,0.8)
    if xPeriod is None:
        xPeriod	= random.randrange(10)
    if yPeriod is None:
        yPeriod	= random.randrange(10)
    if turbPower is None:
        turbPower = random.uniform(0,3)
    if turbSize is None:
        turbSize = 2**random.randrange(2,7)
    if offsetX is None:
        offsetX = -random.randrange(shape[0])
    if offsetY is None:
        offsetY = -random.randrange(shape[0])
    
    noise = np.zeros(shape).astype(np.uint8)
    noise = addNoise(noise)

    img = np.zeros((224,224,3)).astype(np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            xyValue = math.sqrt(((i+offsetX)*xPeriod)**2+((j+offsetY)*yPeriod)**2) + (turbPower * turbulence(i,j, turbSize, noise))
            sineValue = 255 * abs(math.sin(math.radians(xyValue)))**2 * darkness
            img[i,j][0] = sineValue
            img[i,j][1] = sineValue
            img[i,j][2] = sineValue

    return img, {'Darkness':darkness, 'xPeriod':xPeriod, 'yPeriod':yPeriod, 'turbPower':turbPower, 'turbSize':turbSize, 'offsetX':offsetX, 'offsetY':offsetY}

# Creates a random symmetry, returns array with image, its symmetry axis and its label; parameters can be modified.
def createAsymmetry(minst, id1 = None, id2 = None,initialRotation2 = None ,initialRotation1 = None, overFlow = None, padding = None, finalRotation = None, resizingPercent = None):
    # Getting the image and label
    if id1 is None:
        id1 = random.randrange(len(minst))
    if id2 is None:
        id2 = random.randrange(len(minst))
    result, _ = getImageArray(id1,minst)
    result = applyColorGradient(result)
    otherNumber, _ = getImageArray(id2,minst)
    otherNumber = applyColorGradient(otherNumber)
    
    # Initial rotation
    if initialRotation1 is None:
        initialRotation1 = random.randrange(360)
    result,_ = rotateDigit(result, initialRotation1)
    if initialRotation2 is None:
        initialRotation2 = random.randrange(360)
    mirrored,_ = rotateDigit(otherNumber, initialRotation2)

    # Combining initial with the secondNumber 
    if overFlow is None:
        overFlow = bool(random.getrandbits(1))
    if padding is None:
        padding = random.randrange(-result.shape[0], result.shape[0])
    result = addWithPadding(result,mirrored, padding, overFlow=overFlow)

    # Adding padding for rotation
    result = addRotationPadding(result)

    # Final rotation
    if finalRotation is None:
        finalRotation = random.randrange(360)
    result,_ = rotateDigit(result,finalRotation)

    # Remove excess pading
    result, _, _, _ = removePadding(result, (0,0), (0,0), (0,0))

    # Resizing
    if resizingPercent is None:
        resizingPercent = random.randrange(80,300)
    result, _ ,_, _, _, _ = resizeSymmetry(resizingPercent, result, (0,0), (0,0), (0,0), 0, 0)

    return result

# Returns an image with a local symmetry its dictionary and the backgrounds dictionary
def getLocalSymmetry(shape, mnist, numOfSymmetries = None, types = ['simple', 'cross'], weights = [0.5, 0.5], idx = None, initialRotation = None, overFlow = None, padding1 = None,
                     padding2 = None, finalRotation = None, resizingPercent = None, backgroundType = None, darknessBackground = None, xPeriod = None, yPeriod = None, turbPower = None, 
                     turbSize = None, offsetX = None, offsetY = None, inverse = None):

    # Digits 
    if idx is None:
        idx = random.randrange(len(mnist))
    # Looping until real combination is found
    found = False
    while not found:
        try:
            digits, dictSymmetries, numDecoys = getRandomDigitsWithSymmetry(idx, mnist, shape, types, weights, numOfSymmetries, initialRotation, 
                                                                            overFlow , padding1, padding2 , finalRotation, resizingPercent)
            found = True
        except:
            found = False

    # Background
    if backgroundType is None:
        backgroundType = random.randrange(2)
    if backgroundType == 0:
        background, dictBack = getSmoothNoiseSin(shape, darknessBackground, xPeriod, yPeriod, turbPower, turbSize)
        dictBack['offsetX'] = 0
        dictBack['offsetY'] = 0
    else:
        background, dictBack = getSmoothNoiseWood(shape, offsetX, offsetY, darknessBackground, xPeriod, yPeriod, turbPower, turbSize)
    background = applyColorGradient(background)

    # Adding digits and background
    img = addNoOverflow(digits, background)

    # Inverse image
    if inverse is None:
        inverse = random.getrandbits(1)
    if inverse:
        img = (255-img)

    # Modifying dictionaries        
    dictBack['backgroundType'] = backgroundType
    dictBack['numDecoys'] = numDecoys
    dictBack['inverse'] = inverse
    for dictSym in dictSymmetries:
        dictSym['centerX'] = dictSym['center'][0]
        dictSym['centerY'] = dictSym['center'][1]

    return img, dictSymmetries, dictBack

# Creates a rotational symmetry of the desired order
def createRotationalSymmetry(id, mnist, rotAxis = None, order = None, resizingPercent = None, color = True):
    result,label = getImageArray(id,mnist)
    if color:
        result = applyColorGradient(result)

    # Selecting rotation axis
    if rotAxis is None:
        rotAxis = (random.randint(0,result.shape[0]-1),random.randint(0,result.shape[1]-1))

    # Inserting image in bigger canvass
    diagonal = int((result.shape[0]**2 + result.shape[1]**2)**0.5)
    canvass = np.zeros((diagonal*2, diagonal*2, 3)).astype(np.uint8)
    insertPointOnPoint(result, rotAxis, canvass, (diagonal, diagonal))

    # Selecting order
    if order is None:
        order = random.randint(3,10)

    # Performing rotation
    rotatedStep = []
    for i in range(order):
        rotatedStep.append(performRotationStep(canvass, (diagonal, diagonal), order, i))
    result = addAllImages(rotatedStep)

    # Remove padding
    result, _, _, newAxis = removePadding(result, [0,0], [0,0], (diagonal, diagonal))
    newAxis = (newAxis[1],newAxis[0])

    # Resize
    if resizingPercent is None:
        resizingPercent = random.randrange(80,300)
    result, _, resizedAxis, _, _ = resizeSymmetryMultipleAxes(250, result, [], newAxis, 0, 0)

    dict = {
        'rotationAxis': resizedAxis,
        'initialRotationAxis': rotAxis,
        'order': order,
        'resizingPercent': resizingPercent,
        'label': label
    }

    return result, dict

# Returns random rotational asymmetry
def getRandomRotationalAssymetry(mnist, sameDigit = True, digit = None, sameColor = True, order = None):
    # Generating all digits
    if order is None:
        order = random.randint(3,10)
    if digit is None:
        digit = random.randint(0,9)
    images = []
    colorToken = None
    for _ in range(order):
        if sameDigit:
            im = getRandomMNISTDigit(mnist, digit)
        else:
            im,_ = getImageArray(random.randint(0,1000), mnist)
        # Applying color
        if sameColor:
            im, colorToken = applyReplicableColorGradient(im, colorToken)
        else:
            im = applyColorGradient(im)
        images.append(im)
    
    # Obtaining min and max dimensions
    minXY, maxXY = getMinMaxHeightAndWidth(images)

    # Random rotation axis
    rotationAxis = (random.randint(0,minXY[0]-1),random.randint(0,minXY[1]-1))

    # Rotating them
    diagonal = int((maxXY[0]**2 + maxXY[1]**2)**0.5)
    rotatedStep = []
    for i in range(order):
        canvass = np.zeros((diagonal*2, diagonal*2, 3)).astype(np.uint8)
        insertPointOnPoint(images[i], rotationAxis, canvass, (diagonal, diagonal))
        rotatedStep.append(performRotationStep(canvass, (diagonal, diagonal), order, i))

    # Union
    return addAllImages(rotatedStep)

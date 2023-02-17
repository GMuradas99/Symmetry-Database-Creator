### IMPORTS ###
import numpy as np
import cv2
import random
import math

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

# Remove the excess padding from image
def removePadding(img, startAxis, endAxis):
    _,_, minX, maxX, minY, maxY = getSAandBB(img)
    cropped = img[minY:maxY+1, minX:maxX+1]
    newStartX = startAxis[0] - minX
    newStartY = startAxis[1] - minY
    newEndX = endAxis[0] - minX
    newEndY = endAxis[1] - minY

    return cropped, (newStartX,newStartY), (newEndX,newEndY)

#Resizes the image and keypoints according to the selected percent
def resizeSymmetry(percent, img, startAxis, endAxis):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)

    newStartX = startAxis[0] * percent / 100
    newStartY = startAxis[1] * percent / 100
    newEndX = endAxis[0] * percent / 100
    newEndY = endAxis[1] * percent / 100

    return cv2.resize(img, (width, height)), (newStartX,newStartY), (newEndX,newEndY)

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

### MAIN FUNCTIONS ###

# Creates a random symmetry, returns array with image, its symmetry axis and its label; parameters can be modified.
def createSymmetry(id, minst, initialRotation = None, overFlow = None, padding = None, finalRotation = None, resizingPercent = None):
    # Getting the image and label
    result,label = getImageArray(id,minst)
    
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
    startAxis, endAxis, _,_,_,_ = getSAandBB(result)

    # Final rotation
    if finalRotation is None:
        finalRotation = random.randrange(360)
    result,rotationMatrix = rotateDigit(result,finalRotation)

    # Rotating symmetry axis
    rotated = transformKeypoints([startAxis, endAxis], rotationMatrix)
    startAxis = rotated[0]
    endAxis = rotated[1]

    # Remove excess pading
    result, startAxis, endAxis = removePadding(result, startAxis, endAxis)

    # Resizing
    if resizingPercent is None:
        resizingPercent = random.randrange(80,300)
    result, startAxis, endAxis = resizeSymmetry(resizingPercent, result, startAxis, endAxis)

    return result, startAxis, endAxis, {'initialRotation':initialRotation, 'overFlow':overFlow, 'padding':padding, 'finalRotation':finalRotation, 'resizingPercent':resizingPercent}


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


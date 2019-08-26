#Author:Meghana V
#College: JNN College of Engineering
#Dept: Computer Science and Engineering
#Sem: 7th
#Submitted to: TCS HumAIn 2019 Campus Connect

import os
import sys
import numpy as np
import cv2
import traceback
from PIL import Image,  ImageDraw
import pytesseract

#Preprocessing Module
def imagepreprocess(imagePath):
    image = cv2.imread(imagePath)
    gimage= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bimage=cv2.bilateralFilter(gimage, 9, 75, 75)
    himage= cv2.equalizeHist(bimage)
    return himage

# Morphological Processing
def morphologicalProcessing(himage, structElem):
    morphImage=cv2.morphologyEx(himage, cv2.MORPH_OPEN, structElem, iterations=15)
    subimage=cv2.subtract(himage, morphImage)
    ret, threshImage = cv2.threshold(subimage, 0, 255, cv2.THRESH_OTSU)
    return threshImage

#Contour Detection
def contourDetection(timage, threshold1, threshold2,iimage):
    cannyImage = cv2.Canny(timage, threshold1, threshold2)
    cannyImage = cv2.convertScaleAbs(cannyImage)
    dilationStructElem = np.ones((3, 3), np.uint8)
    dimage= cv2.dilate(cannyImage, dilationStructElem, iterations=1)
    contours, hierarchy = cv2.findContours(dimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    approximatedPolygon = None
    for contour in contours:
       
        contourPerimeter = cv2.arcLength(contour, True)
        approximatedPolygon = cv2.approxPolyDP(contour, 0.06*contourPerimeter, closed=True)
        # Quadrilateral Detected
        if(len(approximatedPolygon) == 4):
            break
    
    M=cv2.moments(approximatedPolygon)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    
    finalImage = cv2.drawContours(iimage, [approximatedPolygon], -1, (0,255,0),3)
     
    cv2.circle(finalImage, (cX, cY), 7, (0, 255, 0), -1)
    cv2.imwrite('FinalOutput/21c.png', finalImage[360:405,200:460])
    cv2.putText(finalImage, "Centroid of Plate: ("+str(cX)+", "+str(cY)+")", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return finalImage

#Main module
imagePath='Dataset/21.png'
iimage = cv2.imread(imagePath)
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", iimage)
preprocessedImage=imagepreprocess(imagePath)
openingStructElem = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mproimage=morphologicalProcessing(preprocessedImage, openingStructElem)
finalImage=contourDetection(mproimage, 250, 255,iimage)
cv2.namedWindow("Final Output Image", cv2.WINDOW_NORMAL)
cv2.imshow("Final Output Image", finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join("FinalOutput/21-detected.png"), finalImage)
print(pytesseract.image_to_string(Image.open('FinalOutput/21c.png')))

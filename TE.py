import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform

#Functions used

def resize(img,width=500):
    h ,w ,c = img.shape
    height = int( h / w * width )
    
    Rimg = cv2.resize(img,(width,height))

    return Rimg ,(width,height)

def enhance(img):
    return cv2.detailEnhance(img , sigma_s = 50 , sigma_r = 0.25)

def gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #gray scale

def blur(img):
    return cv2.GaussianBlur(img,(5,5),0)

def edge(img):
    return cv2.Canny( img , 200 , 75 )

def morph_trans(img):
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(img,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    return  dilate , closing

def contour(img1 , img2 , color):
    contours , ret = cv2.findContours( img1 , cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key= cv2.contourArea, reverse= True)
    
    for contour in contours:
        peri = cv2.arcLength( contour , True )
        approx = cv2.approxPolyDP(contour , 0.1 * peri ,True)

        if len(approx==4):
            four_points = np.squeeze(approx)
            break
    
    cv2.drawContours(img2,[four_points],-1,color,3)
    
    return four_points
def orig_four_points(img , pts):
    size = resize(img)[1][0]
    multi = img.shape[1] / size
    four_points_orig = (pts * multi)
    four_points_orig = four_points_orig.astype(int)
    return four_points_orig
    

def wrap_image(img,pts):
    return four_point_transform(img ,pts)

def document_scanner(img):
    rimg , size= resize(img)
    enh_img = enhance(rimg)
    gray_img = gray(enh_img)
    edge_img = edge(gray_img)
    dilate_img , close_img= morph_trans(edge_img)
    four_points=contour(close_img,rimg , (0,0,255))
    four_points_originals = orig_four_points(img,four_points)
    wrap_img = wrap_image(img,four_points_originals)
    cv2.imshow('wrap.png',wrap_img)


def show_all_processed_pics(img):

    document_scanner(img)
    cv2.imshow('Capture.png',img)
    cv2.imshow('Resized',rimg)
    cv2.imshow('Enhanced',enh_img)   
    cv2.imshow('Gray',gray_img)
    cv2.imshow('Edge', edge_img)
    cv2.imshow('Dilate', dilate_img)
    cv2.imshow('Close', close_img)
    cv2.imshow('wrap.png',wrap_img)
    cv2.imshow('Cont', rimg)


img = cv2.imread('worked.png')
document_scanner(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
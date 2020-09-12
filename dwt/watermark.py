# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:25:41 2020

@author: Ritik
"""

import cv2
import pywt
import numpy as np
import crypt
#import net_2

class Components:
    '''It is a list used to store the coeffecients of DWT'''
    Coefficients = []    
    U = None
    S = None
    V = None


class embedding:
    
    """
    :param watermark_path:
    :param ratio:
    :param wavelet:
    :param level:
    """
    def __init__(self, watermark_path="logo.jpg", ratio=0.1, wavelet="haar",level=2):
        '''Watermark image(logo)'''
        self.level = level
        self.wavelet = wavelet
        self.ratio = ratio
        
     
        self.shape_watermark = (cv2.imread(watermark_path,0)).shape
        self.W_components = Components()
        self.img_components = Components()
        self.W_components.Coefficients, self.W_components.U, \
        self.W_components.S, self.W_components.V = self.calculate(watermark_path)

    def calculate(self, img):
        '''
        To calculate the Coefficients (DWT)and SVD components.
        :param img: should be a numpy array or the path of the image.
        '''
        if isinstance(img, str):
            img = cv2.imread(img,0)
            
        Coefficients = pywt.wavedec2(img, wavelet=self.wavelet, level=self.level)
        self.shape_LL = Coefficients[0].shape
        U, S, V = np.linalg.svd(Coefficients[0])
        return Coefficients, U, S, V

    def diag(self, s):
        '''
        To recover the singular values to be a matrix.
        :param s: a 1D numpy array
        '''
        S = np.zeros(self.shape_LL)
        row = min(S.shape)
        S[:row, :row] = np.diag(s)
        return S
   

    def recover(self, name):
        '''
        To recover the image from the svd components and DWT
        :param name:
        '''
        components = eval("self.{}_components".format(name))
        s = eval("self.S_{}".format(name))
        components.Coefficients[0] = components.U.dot(self.diag(s)).dot(components.V)
        return pywt.waverec2(components.Coefficients, wavelet=self.wavelet)
    
   
   
    def watermark(self, img="4_1.jpg", path_save=None,):
        '''
        This is the main function for image watermarking.(cover image)
        :param img: image path or numpy array of the image.
        '''
        if not path_save:
            path_save = "watermarked_" + img
        self.path_save = path_save
        
        self.img_components.Coefficients, self.img_components.U, \
        self.img_components.S, self.img_components.V = self.calculate(img)
        self.embed()
        img_rec = self.recover("img")
        cv2.imwrite(path_save, img_rec)
        print("Embedding successful image saved with name-",path_save)
    
    def embed(self):
        self.S_img = self.img_components.S + self.ratio * self.W_components.S * \
                                             (self.img_components.S.max() / self.W_components.S.max())

class Components2:
    '''It is a list used to store the coeffecients of DWT'''
    Coefficients = []    
    U = None
    S = None
    V = None

class extracting:
    
    """
    :param watermark_path:
    :param ratio:
    :param wavelet:
    :param level:
    """
    def __init__(self, watermark_path="4_2.jpg", ratio=0.1, wavelet="haar",level=2):
        '''Watermark image(logo)'''
        self.level = level
        self.wavelet = wavelet
        self.ratio = ratio
     
        self.shape_watermark = (cv2.imread(watermark_path,0)).shape
        self.W_components = Components2()
        self.img_components = Components2()
        self.W_components.Coefficients, self.W_components.U, \
        self.W_components.S, self.W_components.V = self.calculate(watermark_path)

    def calculate(self, img):
        '''
        To calculate the Coefficients (DWT)and SVD components.
        :param img: should be a numpy array or the path of the image.
        '''
        if isinstance(img, str):
            img = cv2.imread(img,0)
            
        Coefficients = pywt.wavedec2(img, wavelet=self.wavelet, level=self.level)
        self.shape_LL = Coefficients[0].shape
        U, S, V = np.linalg.svd(Coefficients[0])
        return Coefficients, U, S, V

    def diag(self, s):
        '''
        To recover the singular values to be a matrix.
        :param s: a 1D numpy array
        '''
        S = np.zeros(self.shape_LL)
        row = min(S.shape)
        S[:row, :row] = np.diag(s)
        return S
   

    def recover(self, name):
        '''
        To recover the image from the svd components and DWT
        :param name:
        '''
        components = eval("self.{}_components".format(name))
        s = eval("self.S_{}".format(name))
        components.Coefficients[0] = components.U.dot(self.diag(s)).dot(components.V)
        return pywt.waverec2(components.Coefficients, wavelet=self.wavelet)
   
    def watermark(self, img="4_2.jpg", path_save=None):
        '''
        This is the main function for image watermarking.(cover image)
        :param img: image path or numpy array of the image.
        '''
        if not path_save:
            path_save = "watermarked_" + img
        self.path_save = path_save
        self.img_components.Coefficients, self.img_components.U, \
        self.img_components.S, self.img_components.V = self.calculate(img)
        #self.embed()
        #img_rec = self.recover("img")
        #cv2.imwrite(path_save, img_rec)
        #print("Embedding successful image saved with name-",path_save)
    
    
        
     

    def extracted(self, image_path=None, ratio=None, extracted_watermark_path = None):
        '''
        Extracted the watermark from the given image.
        '''
        if not extracted_watermark_path:
            extracted_watermark_path = "watermark_extracted_"+image_path
        if not image_path:
            image_path = self.path_save
        img = cv2.imread(image_path,0)
        img = cv2.resize(img, self.shape_watermark)
        img_components = Components2()
        img_components.Coefficients, img_components.U, img_components.S, img_components.V = self.calculate(img)
        ratio_ = self.ratio if not ratio else ratio
        self.S_W = (img_components.S - self.img_components.S) / ratio_
        watermark_extracted = self.recover("W")
        cv2.imwrite(extracted_watermark_path, watermark_extracted)
        print("Extraction successful image saved with name-",extracted_watermark_path)
        
    


if __name__=='__main__':
    print("-------------WATERMARKING---------------------")
    
    '''create object of class watermarking'''
    #net_2.main()
    while True:
        
        print("Press 1-Embedding , 2-Extracting")
        w=int(input("Enter your choice(1 or 2)-"))
        if w==1:
    
            print("Welcome User,,, watermarking will help you to deliver your work securely")
            x=input("Enter the path of the Cover image--")
            y=input("Enter the path of the watermark(eg. COMPANY LOGO) image--")
            print("[+]..Embeeding in process..")
    
            watermarking=embedding(watermark_path=y,level=3)  
            watermarking.watermark(img=x,path_save=None)
            z=int(input("Do you want to add a UID?(Press 1 for Yes/0-NO)"))
            if z==1:
                crypt.main()
            print("-------Thank you.. you can now transmit the watermarked image-----")
        
        elif w==2:
        
    
            print("Found your illegal copy of work, want to extract logo from it to prove your ownership??")
            logo=input("Enter path of the company logo(For making the extracted logo of same size)-")
            watermarking2=extracting(watermark_path=logo,level=3)
       
    
        
            host=input("Enter path of the host file-")
            watermarking2.watermark(img=host)
            ext=input("Enter path of the watermarked image-")
            watermarking2.extracted(image_path=ext)
            enc=int(input("Have u sent an image with UID?(Press 1-Yes,0-NO)-"))
            if enc==1:
                enc_img=input("Enter the path of the encoded image-")
                crypt.decode(enc_img)
        conti=input("Do you wish to continue?-")
        if conti=="yes":
             continue
        else:
            break
    print("-----Thank you--------")
        
        
        

    
        
        
        
    
   
    
    

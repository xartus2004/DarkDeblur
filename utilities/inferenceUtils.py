import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os 
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
from utilities.customUtils import *
from dataTools.sampler import *
import numpy as np
import cv2
from PIL import Image
from dataTools.dataNormalization import *
import skimage.io as io

class AddGaussianNoise(object):
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel
        
    def __call__(self, tensor):
        sigma = self.noiseLevel/255
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor 
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)


class inference():
    def __init__(self, gridSize, inputRootDir, outputRootDir, modelName, resize = None, validation = None ):
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.gridSize = gridSize
        self.modelName = modelName
        self.resize = resize
        self.validation = validation
        self.unNormalize = UnNormalize()
    


    def inputForInference(self, imagePath, noiseLevel):

        img = Image.open(imagePath) #io.imread(imagePath)/255#
        #print(imagePath)
        #if "_BLUR" in imagePath: 
            
        #resizeDimension =  (1024, 1024) 
        #img = img.resize(resizeDimension)
        #    img.save(imagePath)  

        '''img = np.asarray(img) 
        if self.gridSize == 1 : 
            img = bayerSampler(img)
        elif self.gridSize == 2 : 
            img = quadBayerSampler(img)
        elif self.gridSize == 3 : 
            img = dynamicBayerSampler(img, gridSze)
        img = Image.fromarray(img)'''
        #print("img", img.getextrema())
        if self.resize:
            #resize(256,256)
            transform = transforms.Compose([ transforms.Resize(self.resize, interpolation=Image.BICUBIC) ])
            img = transform(img)

        transform = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd),
                                        #AddGaussianNoise(noiseLevel=noiseLevel)
                                        ])

        testImg = transform(img).unsqueeze(0)
        #print("input",imagePath,self.unNormalize(testImg).max(), self.unNormalize(testImg).min())
        return testImg 

    def saveModelOutput(self, modelOutput, inputImagePath, step = None, ext = ".png"):
        datasetName = inputImagePath.split("/")[-2]
        if step:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
                              "_" + self.modelName + "_" + str(step) + ext
        else:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
                              "_" + self.modelName + ext
        save_image(self.unNormalize(modelOutput[0]), imageSavingPath)
        #print(imageSavingPath)
        #print(inputImagePath,self.unNormalize(modelOutput[0]).max(), self.unNormalize(modelOutput[0]).min())

    

    def testingSetProcessor(self):
        testSets = glob.glob(self.inputRootDir+"*/")
        #print ("DirPath",self.inputRootDir+"*/")
        if self.validation:
            #print(self.validation)
            testSets = testSets[:1]
        #print (testSets)
        testImageList = []
        for t in testSets:
            testSetName = t.split("/")[-2]
            #print("Dir Path",self.outputRootDir + self.modelName  + "/" + testSetName )
            createDir(self.outputRootDir + self.modelName  + "/" + testSetName)
            imgInTargetDir = imageList(t, False)
            testImageList += imgInTargetDir

        return testImageList



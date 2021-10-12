from PIL import Image, ImageFile
import requests
import torch 
from torch import nn
import numpy as np
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import cv2
import PIL
from scipy import ndimage



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = "./utils/batch32_adam.pt"


class TongueSeg(nn.Module):
	def __init__(self): 
                super(TongueSeg, self).__init__() 
                self.model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=16384)
	def forward(self, values): 
                outL = self.model(values) 
                return outL


class TongueProcess():
	def __init__(self): 
		self.preprocess = transforms.Compose([ 
			transforms.Resize(256), 
			transforms.CenterCrop(224), 
			transforms.ToTensor(), 
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
			]) 
		self.model = TongueSeg()
		self.model = self.model.to(device) 
		self.model.load_state_dict(torch.load(MODEL, map_location=device))
		self.model = self.model.eval()
		self.l, self.w = 0, 0

	def getLen1d(self, arr):
		if arr.max() == arr.min():
			return 0,0,0
		start = np.argmax(arr)
		end = arr.shape[0] - np.argmax(arr[::-1]) - 1
		l = end - start + 1
		return start, end, l

	def getMax2d(self, arr):
		d = { "index":0, "start":0, "end":0, "length":0, "center":0}
		for i, a in enumerate(arr):
			s, e, l = self.getLen1d(a)
			if l >= d["length"]:
				d['index'] = i
				d['start'] = s
				d['end'] = e
				d['length'] = l
				d['center'] = int(l/2) + s
		return d

	def getDimTongue(self, segImage):
		'''
                Get the info about the length and width of the segmented tongue in the image

		Args:
			segImage (numpy.ndarray): the output segmented mask returned from model. it is grayscale(0-255).

		Outputs:
			dictLength (Dict): info about the length 
			dictWidth (Dict): info about the width
		'''
		dictLength = self.getMax2d(segImage)
		dictWidth = self.getMax2d(np.transpose(segImage))
		return dictLength, dictWidth

	def getMaskRegion(self, segImage):
		''' 
		Get the mask of every region of the tongue.
		
		Args: 
			segImage (numpy.ndarray): the output segmented mask returned from model. it is grayscale(0-255). 

		Outputs: 
			regions (Dict): dictionary has mask of range(0-255) for every regions of the tongue. 
		'''
		self.l,self.w = self.getDimTongue(segImage) 
		infoImages = { 
			#"region":[center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness] 
			"stomach":[(self.l['center'],self.w['center']), (int(self.l['length']*0.2),int(self.w['length']*0.2)), 0, 0, 360, 255, -1], 
			"heart":[(self.l['center'],self.w['end']), (int(self.l['length']*0.3),int(self.w['length']*0.3)), 0, 180, 360, 255, -1], 
			"lungs":[(self.l['center'],self.w['end']), (int(self.l['length']*0.4),int(self.w['length']*0.4)), 0, 180, 360, 255, -1], 
			"kidney":[(self.l['center'],self.w['start']), (int(self.l['length']*0.3),int(self.w['length']*0.2)), 180, 180, 360, 255, -1], 
			"heartDash":[(self.l['center'],self.w['end']), (int(self.l['length']*0.3),int(self.w['length']*0.3)), 0, 180, 360, 0, -1], 
			"kidneyDash":[(self.l['center'],self.w['start']), (int(self.l['length']*0.3),int(self.w['length']*0.5)), 180, 180, 360, 255, -1], 
			"stomachDash":[(self.l['center'],self.w['center']), (int(self.l['length']*0.3),int(self.w['length']*0.3)), 0, 0, 360, 255, -1], 
			}
		
		blank = np.zeros((128,128), dtype='uint8') 
		white = np.ones((128,128), dtype='uint8') 
		white = white * 255 
		heartMaskDash = cv2.ellipse(white.copy(), *infoImages['heartDash']) 
		lungsMaskDash = cv2.ellipse(blank.copy(), *infoImages['lungs']) 
		kidneyMaskDash = cv2.ellipse(blank.copy(), *infoImages['kidneyDash']) 
		stoMaskDash= cv2.ellipse(blank.copy(), *infoImages['stomachDash']) 
		liverMaskDash = cv2.bitwise_or(cv2.bitwise_or(stoMaskDash,lungsMaskDash),kidneyMaskDash) 
		liverMaskDash = np.array(PIL.ImageOps.invert(Image.fromarray(liverMaskDash)))
		liverMask = cv2.bitwise_and(segImage,liverMaskDash)
		kidneyMask= cv2.ellipse(blank.copy(), *infoImages['kidney']) 
		lungsMask = cv2.bitwise_and(heartMaskDash, lungsMaskDash)
		heartMask = cv2.ellipse(blank.copy(), *infoImages['heart']) 
		stoMask= cv2.ellipse(blank.copy(), *infoImages['stomach']) 
		
		return {"liver":liverMask, "kidney":kidneyMask, "lungs":lungsMask, "heart":heartMask, "stomach":stoMask}

	def getColor1d(self, segImage):
		value, num = 0, 0
		unique, counts = np.unique(segImage, return_counts=True)
		dictPixel = dict(zip(unique,counts))
		for key in dictPixel.keys():
			if key > 50 and key < 210:
				value = value + key * dictPixel[key]
				num = num + dictPixel[key]
		return int(value / num)

	def getColor2d(self, segImage):
		rgb = []
		for i in range(0,3):
			rgb.append(self.getColor1d(segImage[:,:,i]))
		return rgb

	def getColorRegions(self, resNP, outMask):
		'''
		Get the color for every region in the tongue
		
		Args:
			resNP (numpy.ndarray): array of the result of multiplication of mask and the input colored image of tongue.
			outMask (numpy.ndarray): the output segmented mask returned from model. it is binary scale(0-1)
		'''
		colorDict = {}
		d = self.getMaskRegion(outMask*255)
		for k in d.keys(): 
			mask = d[k].reshape((128,128,1)) / 255 
			res = np.multiply(resNP, mask) 
			res = np.round(res) 
			res = res.astype(np.uint8)
			color = self.getColor2d(res)
			colorDict[k] = color
		return colorDict

	def getShapeRegions(self, resNP): 
		im = Image.fromarray(resNP).crop((self.l['start'] + 10, self.w['start']+15, self.l['start']+self.l['length']-15, self.w['start']+self.w['length']-10))
		var = ndimage.variance(np.array(im))
		shape = "normal" if var < 400 else "cracks"
		return shape

	def getTongueInfo(self, img):
		'''
		Get color of each region and get shape of the tongue after geting mask from the model.

		Args:
			img (image): image of the tongue of the user.
		
		Outputs:
			tongueDict (Dict): dictionary has color of each region and shape of the tongue.
		'''
		inputs = self.preprocess(img) 
		inputs = inputs.to(device) 
		sig = nn.Sigmoid() 
		outMask = sig(self.model(inputs.unsqueeze(dim=0)))
		outMask = outMask.squeeze().reshape([128,128]) 
		with torch.no_grad(): 
			outMask = outMask.to('cpu').numpy() 
		outMask = np.round(outMask) 
		outMask = outMask.astype(np.uint8)
		img = img.resize((128, 128))
		realNP = np.asarray(img) 
		resNP = np.multiply(realNP,outMask.reshape((128,128,1))) 
		resNP = resNP.astype(np.uint8)
		tongueDict = self.getColorRegions(resNP, outMask)
		shape = self.getShapeRegions(resNP)
		tongueDict['shape'] = shape

		return tongueDict



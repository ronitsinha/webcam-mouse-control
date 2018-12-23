import cv2
import numpy as np

average_color1 = None
average_color2 = None

# Reference to use for background subtraction
bg_ref = np.zeros((480, 848), np.uint8)


cap = cv2.VideoCapture(0)

def background_subtraction (gray, thresh, out):
	img = gray.copy()

	diff = np.absolute( np.subtract (img, bg_ref) )

	print diff
	out[diff <= thresh] = 0

# def furthest_defects (defects, center, num, largest):
# 	def_sort = defects.sort(key=lambda d: (d[0]-center[])**2 + ()**2)

# 	def_sort

	
# 	return def_sort[:num]

def closest_defects (defects, center, num):
	def_sort = defects.sort()

	
	return def_sort[:num]


while True:
	ret, frame = cap.read()

	frame = cv2.flip(frame, 1)
	raw = frame.copy()

	if bg_ref.all()== 0:
		bg_ref = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
	
	
	gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
	imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	colorRegion1 = imgHSV[100:300, 450:525]
	cv2.rectangle(frame,(450,100),(525,300),average_color1 or (255,0,0),3)

	colorRegion2 = imgHSV[100:300, 525:600]
	cv2.rectangle(frame,(525,100),(600,300),average_color2 or (255,0,0),3)


	hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)


	if average_color1 and average_color2:
		minH = min(average_color2[0], average_color1[0]) 
		minS = min(average_color2[1], average_color1[1]) 
		minV = min(average_color2[2], average_color1[2]) 

		maxH = max(average_color2[0], average_color1[0]) 
		maxS = max(average_color2[1], average_color1[1]) 
		maxV = max(average_color2[2], average_color1[2])
		
		mask = cv2.inRange(hsv, (minH-50, minS-50, 0), (maxH+50, maxS+50, 255))

		kernel = np.ones((6,6),np.uint8)
		opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		dilation = cv2.dilate(mask,np.ones((1,1),np.uint8),iterations = 1)

		
		# Find the largest contour (ideally the hand)
		_, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		if contours:

			largest = max(contours, key=cv2.contourArea)

			x,y,w,h = cv2.boundingRect(largest)
			hull = cv2.convexHull(largest,returnPoints = False)
			defects = cv2.convexityDefects(largest,hull)

			if not defects is None:


				for i in range(defects.shape[0]):
				    s,e,f,d = defects[i,0]
				    start = tuple(largest[s][0])
				    end = tuple(largest[e][0])
				    far = tuple(largest[f][0])
				    cv2.line(frame,start,end,[0,255,0],2)
				    cv2.circle(frame,far,5,[0,0,255],-1)

			# cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

		_, thresh1 = cv2.threshold( cv2.absdiff(bg_ref, gray), 5, 255,  cv2.THRESH_BINARY )

		# background_subtraction(gray, 10, mask)
		cv2.imshow('mask', mask)

	cv2.imshow('hsv', hsv)

	cv2.imshow('raw', frame)

	k = cv2.waitKey(1)

	if k == ord('r'):
		print('R key pressed!')

		average_color1 = [colorRegion1[:, :, i].mean() for i in range(colorRegion1.shape[-1])]
		average_color2 = [colorRegion2[:, :, i].mean() for i in range(colorRegion1.shape[-1])]
	elif k == 27:
		break

cap.release()
cv2.destroyAllWindows()
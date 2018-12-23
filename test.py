import cv2
import numpy as np
import math

from pymouse import PyMouse

m = PyMouse()

screen_w, screen_h = m.screen_size()


average_color = None
bg_ref = np.zeros((480, 640))
mouse_ready = False;
move_mouse = False;


# subtract face detection from mask -- improves color-based hand tracking
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def background_subtraction (gray, thresh, out):
	img = gray.copy()

	diff = np.absolute( np.subtract (img, bg_ref) )

	print diff
	out[diff <= thresh] = 0

while True:
	ret, raw = cap.read()
	raw = cv2.flip(raw, 1)

	frame =  cv2.blur(raw,(35,35)) #raw.copy()
	face_mask = np.zeros ((480, 640),np.uint8)


	gray_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if bg_ref.all() == 0:
		bg_ref = gray
		
	colorROI = hsv[100:300, 450:600]
	cv2.rectangle(frame,(450,100),(600,300),average_color or (255,0,0),3)

	if average_color:
		mask = cv2.inRange(hsv, (average_color[0]-20, average_color[1]-20, 0), (average_color[0]+20, average_color[1]+20, 255))


		faces = face_cascade.detectMultiScale(gray_raw, 1.3, 5)

		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			face_mask[y:y+h, x:x+w] = cv2.inRange(hsv[y:y+h, x:x+w], (average_color[0]-20, average_color[1]-20, 0), (average_color[0]+20, average_color[1]+20, 255))



		bg_sub = np.absolute( np.subtract (gray, bg_ref) )

		mask = np.subtract(mask, face_mask)
		mask[bg_sub < 5] = 0


		erosion = cv2.erode(mask,np.ones((3,3),np.uint8),iterations = 1)
		dilation = cv2.dilate(erosion,np.ones((6,6),np.uint8),iterations = 1)

		res = cv2.bitwise_and (raw, raw, mask=mask)

		# Find the largest contour (ideally the hand)
		_, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		if contours:

			largest = max(contours, key=cv2.contourArea)

			x,y,w,h = cv2.boundingRect(largest)
			center_x, center_y = x+w/2, y+h/2
			cv2.circle(frame, (center_x, center_y), 7, (255, 255, 255), -1)


			# Get center point for potential mouse movement
			scaled_x = math.floor(( float(center_x) /frame.shape[1])*screen_w)
			scaled_y = math.floor(( float(center_y) /frame.shape[0])*screen_h)

			if move_mouse:
				m.move(scaled_x, scaled_y)

			hull = cv2.convexHull(largest,returnPoints = False)
			defects = cv2.convexityDefects(largest,hull)

			far_x  = np.array([])
			far_y = np.array([])

			if not defects is None:
				for i in range(defects.shape[0]):
					s,e,f,d = defects[i,0]
					start = tuple(largest[s][0])
					end = tuple(largest[e][0])
					far = tuple(largest[f][0])
					cv2.line(frame,start,end,[0,255,0],2)
					cv2.circle(frame,far,5,[0,0,255],-1)


					far_x = np.append(far_x, far[0])
					far_y = np.append(far_y, far[1])
					# if far[1] < center_y :
					cv2.line(frame,start,end,[0,255,0],2)
					cv2.circle(frame,far,5,[0,0,255],-1)
						

			# TODO: use either kmeans algorithm or something else to find "fingers" on hand

			# far_k = np.float32( np.vstack((far_x, far_y)) )
			# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
			# ret,label,center=cv2.kmeans(far_k,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
			
			# if center is not None:
			# 	for i in range(len(center)):
			# 		if len(center[i]) >= 2:
			# 			print(center[i, 0])
			# 			print(str(center[i, 1]) + "\n\n")
			# 			cv2.circle(frame,(center[i,0], center[i, 1]),5,[255,0,0],-1)

		cv2.imshow('faces', face_mask)
		cv2.imshow('mask', res)

	# cv2.imshow('hsv', hsv)
	cv2.imshow('blur', frame)
	cv2.imshow('raw', raw)

	k = cv2.waitKey(1)

	if k == ord('r'):
		print ("Average color taken!")
		average_color = [colorROI[:, :, i].mean() for i in range(colorROI.shape[-1])]
		mouse_ready = True;
		print ("Press 'T' to toggle mouse tracking")
	if k == ord('t') and mouse_ready:
		move_mouse = not move_mouse;
		print('Mouse movement toggled ' + ('ON' if move_mouse else 'OFF') );
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
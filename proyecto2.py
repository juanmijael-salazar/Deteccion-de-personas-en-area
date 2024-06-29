import cv2
import numpy as np

cap = cv2.VideoCapture('video0.mp4')


fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


while True:

	ret, frame = cap.read()
	if ret == False: break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Dibuja un rectángulo en frame, para señalar el objeto
	cv2.rectangle(frame,(0,0),(frame.shape[1],0),(0,0,0),-1)
	color = (0, 255, 0)
	texto_estado = "Estado: No se ha detectado movimiento"

	# Especificamos los puntos extremos del área a analizar
	area_pts = np.array([[1430,170], [630,140],[320,850],       [1830,1000]])
	#area_pts = np.array([[1250,40], [900,40],[390,320], [60,940], [600,900], [200,900], [0,frame.shape[0]], [1000,frame.shape[2]]])


	# Genera imagenes auxiliares
	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)

	image_area = cv2.bitwise_and(gray, gray, mask=imAux)
	fgmask = fgbg.apply(image_area)

	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	fgmask = cv2.dilate(fgmask, None, iterations=10)


	#Encontramos los contornos presentes en fgmask
	cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	for cnt in cnts:
		if cv2.contourArea(cnt) > 200:
			x, y, w, h = cv2.boundingRect(cnt)
			cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 3)
			texto_estado = "Estado: Alerta Movimiento Detectado!"
			color = (0, 0, 255)

	#Visuzalizamos el contorno del área que vamos a analizar y el estado de la detección de movimiento
	cv2.drawContours(frame, [area_pts], -1, color, 10)
	cv2.putText(frame, texto_estado , (10, 30),
			   cv2.FONT_HERSHEY_SIMPLEX, 1, color,3)

	cv2.imshow('fgmask', fgmask)
	cv2.imshow("frame", frame)

	k = cv2.waitKey(70) & 0xFF
	if k == 27:
		break


cap.release()
cv2.destroyAllWindows()

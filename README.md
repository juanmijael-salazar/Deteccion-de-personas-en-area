# Detección de personas para el control y monitoreo en áreas verdes y parques infantiles
# Métodos usados
Se usa **BackgroundSubtractorMOG**. Sustracción de fondo
```
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
```

Transformar a escala de grises
```
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

Dibujado de área a capturar movimiento
```
cv2.rectangle(frame,(0,0),(frame.shape[1],0),(0,0,0),-1)
color = (0, 255, 0)
```
Puntos de vértice para dibujar el área
```area_pts = np.array([[1430,170], [630,140],[320,850],       [1830,1000]])```
o
```area_pts = np.array([[1250,40], [900,40],[390,320], [60,940], [600,900], [200,900], [0,frame.shape[0]], [1000,frame.shape[2]]])```


Genera imágenes auxiliares para determinar el área donde se va a hacer la detección de
movimiento
```
imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)

image_area = cv2.bitwise_and(gray, gray, mask=imAux)
fgmask = fgbg.apply(image_area)
```
Se aplican transformaciones morfológicas para mejorar la imagen binaria obtenida 
```
fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
fgmask = cv2.dilate(fgmask, None, iterations=10)
```

Se usa **cv2.findContours** para encontrar los contornos presentes

```cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]```


Y para poder determinar si existe movimiento
```
	for cnt in cnts:
		if cv2.contourArea(cnt) > 200:
			x, y, w, h = cv2.boundingRect(cnt)
			cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 3)
			texto_estado = "Estado: Alerta Movimiento Detectado!"
			color = (0, 0, 255)
```

Vértices para área: ```area_pts = np.array([[1430,170], [630,140], [320,850], [1830,1000]])```
![](https://github.com/juanmijael-salazar/Deteccion-de-personas-en-area/blob/main/vision%20comp%20images/VC1.png)

Vértices para área: ```area_pts = np.array([[1250,40], [900,40], [390,320], [60,940], [1680,920]])```
![](https://github.com/juanmijael-salazar/Deteccion-de-personas-en-area/blob/main/vision%20comp%20images/VC2.png)

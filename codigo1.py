import cv2

loadAlgort = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

imagem = cv2.imread("fotosocv/foto2.jpg")

# Transformando a imagem em cinza (para o modelo entender de forma mais facil)
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# scaleFactor diminui a escala
# minNeighbors ajuste de parametros (manual e pode variar de uma imagem para a outra)
# minSize ajusta o tamanho minimo da deteccao da face
faces = loadAlgort.detectMultiScale(imagemCinza, scaleFactor=1.08, minNeighbors=7, minSize=(30,30))

print(faces)

for(x, y, l, a) in faces:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 0, 255), 2)

cv2.imshow("Faces", imagem)
cv2.waitKey()

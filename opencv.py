import cv2

# Carregando o algoritimo da pasta
loadAlgort = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Criando variavel para salvar a imagem
imagem = cv2.imread("fotosocv/foto3.jpg")

# Transformando a imagem em cinza (para o modelo entender de forma mais facil)
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = loadAlgort.detectMultiScale(imagemCinza)

print(faces)

# Colocar marcação em verde ao encontrar um rosto
for(x, y, l, a) in faces:
    cv2.rectangle(imagem, (x,y), (x + l, y + a), (0,255,0), 2 )

cv2.imshow("Faces", imagem)
cv2.waitKey()

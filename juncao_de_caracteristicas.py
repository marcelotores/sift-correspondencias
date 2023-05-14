import cv2 as cv
import numpy as np

imagem = 'imagens/1.jpg'

img1 = cv.imread(imagem, cv.IMREAD_GRAYSCALE)

blur = cv.blur(img1, (3, 3))
GaussianBlur = cv.GaussianBlur(img1, (5, 5), 0)
medianBlur = cv.medianBlur(img1, 7)
bilateralFilter = cv.bilateralFilter(img1, 9, 75, 75)

# Função imprime
def imprime_informacoes(kp):
  if len(kp) == 0:
    print('Não há pontos')
    return
  print('Quantidade de Pontos: ', len(kp))
  print('Coordenadas')
  for i in range(len(kp)):
    print(kp[i], ' - ', kp[i].pt)

  print('Relevância')
  for i in range(len(kp)):
    print(kp[i], ' - ', kp[i].response)

img_sem_filtro = img1
sift = cv.SIFT_create()
kp = sift.detect(img_sem_filtro, None)


# Calculando N características mais relevantes e Calculando descritores das características mais relevantes
n = 4
kp_relevantes = sorted(kp, key=lambda x: -x.response)[:n]

# Calculando descritores
kp, des = sift.compute(img_sem_filtro, kp_relevantes)

nova_img = cv.drawKeypoints(img_sem_filtro, kp, img_sem_filtro)

cv.imshow('Imagem sem filtro', nova_img)
imprime_informacoes(kp)
cv.waitKey(0)

## Imagem com Filtro

#img_com_filtro = blur
#img_com_filtro = GaussianBlur
img_com_filtro = medianBlur
img_com_filtro2 = medianBlur2
#img_com_filtro = bilateralFilter

kp2 = sift.detect(img_com_filtro, None)

# Calculando N características mais relevantes e Calculando descritores das características mais relevantes
n = 3
kp_relevantes2 = sorted(kp2, key=lambda x: -x.response)[:n]
#print(kp_relevantes[1].pt)

# Calculando descritores
kp2, des2 = sift.compute(img_com_filtro, kp_relevantes2)
#print(kp[1].pt)

nova_img = cv.drawKeypoints(img_com_filtro, kp2, img_com_filtro)

cv2_imshow(nova_img)

imprime_informacoes(kp2)

print(des2)



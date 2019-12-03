import cv2, numpy as np
def printImagen(im):
    cv2.imshow('imagen',im);cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)


image = cv2.imread('test.png')
printImagen(image)
image = np.float32(image)

gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
print(gx)
printImagen(gx)
printImagen(gy)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
printImagen(mag)

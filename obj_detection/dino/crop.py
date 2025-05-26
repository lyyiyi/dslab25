import cv2

p = "/work/courses/dslab/team14/training/hand/IMG_7209.jpg"
new_p = "/work/courses/dslab/team14/training/hand/hand.jpg"

img = cv2.imread(p)

print(img.shape)

img = img[:2800, 500:-500]

cv2.imwrite(new_p, img)
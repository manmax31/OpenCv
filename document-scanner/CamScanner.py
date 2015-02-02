__author__ = 'manabchetia'

from pyimagesearch import imutils
import cv2


def resize(image):
    height = 500
    ratio = image.shape[0] / height
    image = imutils.resize(image, height)
    return image, ratio


def main():
    # Read Image
    image_path = "/Users/manabchetia/Documents/PyCharm/OpenCV/document-scanner/images/page.jpg"
    orig_image = cv2.imread(image_path)

    # Resize Image
    image = orig_image.copy()
    image, ratio = resize(image)

    # Gray Image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray_image, 75, 200)

    # Show Image
    cv2.imshow("Image", image)
    cv2.imshow("Edge Image", edge_image)
    cv2.waitKey(0)


if __name__ == '__main__': main()





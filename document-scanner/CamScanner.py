__author__ = 'manabchetia'

from pyimagesearch import imutils
from pyimagesearch.transform import four_point_transform
from skimage.filter import threshold_adaptive
import cv2


def resize(image):
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)
    return image, ratio


def get_contours(edge_image):
    contours, _ = cv2.findContours(edge_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    return contours


def get_4_corners(edge_image):
    contours = get_contours(edge_image)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            corners_4 = approx
            break

    page = cv2.convexHull(corners_4)

    return page


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

    # Get 4 corners of page
    page = get_4_corners(edge_image)
    cv2.drawContours(image, [page], -1,  (0, 255, 0), 2)

    # Get top-down view of original image using 4 point transform
    top_view_image = four_point_transform(orig_image, page.reshape(4, 2) * ratio)
    top_view_image = cv2.cvtColor(top_view_image, cv2.COLOR_BGR2GRAY)
    top_view_image = threshold_adaptive(top_view_image, 255, offset=10)
    scanned_image  = top_view_image.astype("uint8") * 255


    cv2.imshow("Scanned image", imutils.resize(scanned_image, height=650))
    cv2.waitKey(0)



if __name__ == '__main__': main()





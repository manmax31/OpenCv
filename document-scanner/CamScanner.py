__author__ = 'manabchetia'

from libs import imutils
from libs.transform import four_point_transform
from skimage.filter import threshold_adaptive
import cv2


def resize(image):
    """
    Returns resized image
    :param image: Original RGB image
    :return image: resized image
    :return ratio: ratio by which it is resized
    """
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)
    return image, ratio


def get_contours(edge_image):
    """
    Returns the top 4 largest contours
    :param edge_image: image containing edge information
    :return contours: largest 4 contours
    """
    contours, _ = cv2.findContours(edge_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    return contours


def get_4_corners(edge_image):
    """
    Returns the contours of paper
    :param edge_image: image containing edge information
    :return page: contour information of the page
    """
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
    """
    Execution begins here
    """

    # Read Image
    image_path = "/Users/manabchetia/Documents/PyCharm/OpenCV/document-scanner/images/receipt.JPG"
    orig_image = cv2.imread(image_path)

    # Resize Image
    image = orig_image.copy()
    image, ratio = resize(image)
    cv2.imshow("Orig Image", image)

    # Gray Image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray_image, 75, 200)

    # Get 4 corners of page
    page = get_4_corners(edge_image)
    cv2.drawContours(image, page, -1,  (0, 255, 0), 2)

    # Get top-down view of original image using 4 point transform
    top_view_image = four_point_transform(orig_image, page.reshape(4, 2) * ratio)
    top_view_image = cv2.cvtColor(top_view_image, cv2.COLOR_BGR2GRAY)
    top_view_image = threshold_adaptive(top_view_image, 255, offset=10)
    scanned_image  = top_view_image.astype("uint8") * 255

    cv2.imshow("Scanned image", imutils.resize(scanned_image, height=650, width=650))
    cv2.waitKey(0)

if __name__ == '__main__': main()





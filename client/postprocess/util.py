import cv2
import numpy as np

def get_min(x_coords, appear_num=5):
    x1 = np.min(x_coords)
    min_count = np.count_nonzero(x_coords == x1)
    # Find the minimum value that appears at least five times
    while min_count < appear_num:
        x1 = np.min(x_coords[x_coords > x1])
        min_count = np.count_nonzero(x_coords == x1)
    return x1


def convert_points_to_xywh(points):
    """
    Convert bbox from format 4 points to (x,y,w.h)
    """
    # Assuming points is a list of four points in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    x_values, y_values = zip(*points)
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    x = min_x
    y = min_y
    w = max_x - min_x
    h = max_y - min_y

    x, y, w, h = int(x), int(y), int(w), int(h)

    return x, y, w, h


def create_color_mask(image, color="saturation", show=False):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask: np.array
    if color == "saturation":
        # lower mask (0-10)
        lower_color = np.array([0, 0, 0])
        upper_color = np.array([180, 200, 255])
        mask = cv2.inRange(img_hsv, lower_color, upper_color)
    else:
        raise NotImplementedError

    return mask


def get_mask_for_fit_text(text_image, invert, remove_noise_in_background):
    # ------ Step 1: text mask
    # Way 1: contrast + gray image + blur + threshold
    # a) augment color (contrast)
    image_aug = cv2.convertScaleAbs(text_image, alpha=1.2, beta=0)
    # b) Convert the image to grayscale
    gray = cv2.cvtColor(image_aug, cv2.COLOR_BGR2GRAY)
    # c) remove noise
    color_mask = None
    if remove_noise_in_background:
        # remove 1:
        gray = cv2.medianBlur(gray, 5)
        # remove 2: color mask
        color_mask = create_color_mask(text_image, color="saturation")
    # d) Threshold
    if invert:
        thresholded = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
        )
    else:
        thresholded = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
        )

    if color_mask is not None:
        text_mask = cv2.bitwise_and(thresholded, color_mask)
    else:
        text_mask = thresholded
    return text_mask


def fit_margin_of_text_2(
    text_image,
    text_mask=None,
    remove_noise_in_background=False,
    invert=True,
    show=False,
):
    """ [new] remove tiny blob in top
    invert: text is black, no invert: text is white
    """
    # --- default
    if text_mask is None:
        text_mask = get_mask_for_fit_text(text_image, invert, remove_noise_in_background)

    # ------ Step 2: bbox
    y_coords, x_coords = np.where(text_mask > 0)
    h, w = text_mask.shape[:2]
    appear_num_x = min(5, 0.025*h)
    appear_num_y = min(5, 0.025*w)
    if y_coords.size != 0 and x_coords.size != 0:
        x1 = np.min(x_coords)
        y1 = get_min(y_coords, appear_num=appear_num_y)
        x2 = np.max(x_coords)
        y2 = np.max(y_coords)
        # convert numpy.int64 to int
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)      
    else:
        x1, y1, x2, y2 = None, None, None, None     # cannot fit text_image
    if show:
        img_draw = text_image.copy()
        color = (0, 255, 0)  # Green color in BGR
        thickness = 2
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        
        # Display the image with the rectangle
        cv2.imshow("Rectangle", img_draw)
        cv2.imshow('mask', text_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return y1, x1, y2, x2


def fit_bbox_2(image, bbox, type="4_points"):
    if not bbox:
        return None
    
    if type == "4_points":
        x1 = int(min([point[0] for point in bbox]))
        y1 = int(min([point[1] for point in bbox]))
        x2 = int(max([point[0] for point in bbox]))
        y2 = int(max([point[1] for point in bbox]))
    else:
        (x1, y1, x2, y2) = bbox

    top, left, bot, right = fit_margin_of_text_2(
        image[y1:y2, x1:x2].copy()
    )  # y1, x1, y2, x2
    if None in (top, left, bot, right):     # cannot fit text_image
        return None
    # fit rectangle
    x2 = x1 + right
    y2 = y1 + bot
    x1 = x1 + left
    y1 = y1 + top

    return (x1, y1, x2, y2)


def fit_bboxes_2(image, bboxes):
    fit_rects = []
    for bbox in bboxes:
        fitted_bbox = fit_bbox_2(image, bbox)
        if fitted_bbox is not None: # fit bbox
            (x1, y1, x2, y2) = fitted_bbox
        else:                       # Not fit bbox
            x, y, w, h = convert_points_to_xywh(bbox)
            (x1, y1, x2, y2) = x, y, x+w, y+h
        fit_rects.append([x1, y1, x2, y2])
    return fit_rects
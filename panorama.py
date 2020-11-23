import os, os.path
import cv2
import numpy as np

def getStitchedImage(img1, img2, M):
    '''
    function to compute stitched img based on two input imgs anda given homography matrix
    Input:
        two images
        a calculated hography matrix
    Output:
        a stitched image 
    '''
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Getting the canvas dimesions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Getting relative perspective of the second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Calculating dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Creating output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))
    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] = img1

    # All done now just return img :)
    return result_img



def getSiftHomography(img1, img2):
    '''
    function to compute homography matrix using SIFT
    Input:
        two images
    Output:
        a calculated homography matrix
    '''

    sift = cv2.xfeatures2d.SIFT_create()

    # Extract keypoints and descriptors
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)


    verify_ratio = 0.8  # sauce: dont ask pls
    verified_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)

    min_matches = 8
    if len(verified_matches) > min_matches:

        img1_pts = []
        img2_pts = []

        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # Computing homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M
    else:
        orint("Not enough matches")
        return False

def equalizeHistogram(img):
    '''
    function to equalize histogram
    Input:
        an image
    Output:
        an image after being equalized
    '''

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

def getImage():
    '''
    function to init neccesary images
    Input:
        none, we use static url kekw
    Output:
        set of images with histogram equalization
    '''
    path = 'inputs' #feel free to change this to ur img folder

    temps = []
    allowed_ext = [".jpg", ".gif", ".png", ".tga", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in allowed_ext:
            continue
        temps.append(cv2.imread(os.path.join(path, f)))

    imgs = []
    for i in temps:
        imgs.append(equalizeHistogram(i))

    return imgs

def panorama():
    '''
    main function
    '''
    imgs = getImage()
    result_image = imgs[0]
    for i in range(1, len(imgs)):
        M = getSiftHomography(result_image, imgs[i])
        result_image = getStitchedImage(imgs[i], result_image, M)

    cv2.imwrite('result.jpg', result_image)
    if result_image.shape[1]<1500:
        cv2.imshow('result',result_image)

def main():
    '''
    just a main function to call main function idk why
    '''
    panorama()

if __name__ == '__main__':
    '''
    ... k why we need this?
    '''
    main()
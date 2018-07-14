import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys
from pprint import pprint

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_color = cv2.imread("images/pattern001.jpg")
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape
    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)
    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2)), (0,0), fx=scale_factor,fy=scale_factor)
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE)/255.0, (0,0),fx=scale_factor, fy=scale_factor)
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        i=0
        while i<on_mask.shape[0]:
            j=0
            while j<on_mask.shape[1]:
                if on_mask[i][j]:
                    scan_bits[i][j] = scan_bits[i][j] + bit_code
                j+=1
            i+=1

        # for every point in mask where mask is on assign the bit_code value to scan bits
        # each point in a scan bit is a 16 bit binary code.
        # based on the image number we have to just change that bit
        # TODO: populate scan_bits by putting the bit_code according to on_mask

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","rb") as f:
        binary_codes_ids_codebook = pickle.load(f)
        # u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        # binary_codes_ids_codebook = u.load()

    camera_points = []
    projector_points = []
    # print(h)
    corr_img = np.zeros((proj_mask.shape[0], proj_mask.shape[1], 3))
    RGB_value = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code
            else:
                x_p,y_p = binary_codes_ids_codebook[scan_bits[y,x]]
                if x_p >= 1279 or y_p >= 799:  # filter
                    continue
                else:
                    # data[y,x] = [0,y_p/2,x_p/2]
                    camera_points.append([[x/2, y/2]])
                    projector_points.append([[x_p , y_p]])
                    corr_img[y, x, 2] = np.uint8((x_p / 1280.0) * 255)
                    corr_img[y, x, 1] = np.uint8((y_p / 960.0) * 255)
                    RGB_value.append(ref_color[y,x])

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
    RGB_value=np.asarray(RGB_value)
    # print(RGB_value.shape)
    # now that we have 2D-2D correspondances, we can triangulate 3D points!
    cv2.imwrite("correspondance.png", np.array(corr_img).astype('uint8'))

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","rb") as f:
        d = pickle.load(f)
        # u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        # d = u.load()
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']


    # TODO: use    to get normalized points for camera, use camera_K and camera_d
    nCameraPoints = cv2.undistortPoints(np.asarray(camera_points,dtype=np.float32),camera_K,camera_d)
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    nProjectorPoints = cv2.undistortPoints(np.asarray(projector_points,dtype=np.float32), projector_K, projector_d)
    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    prj1 = np.eye(4)[:3]
    prj2 = np.concatenate((projector_R,projector_t), axis=1)
    triangulatePoints = cv2.triangulatePoints(prj1,prj2,nCameraPoints,nProjectorPoints)
    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    real3D = cv2.convertPointsFromHomogeneous(np.transpose(triangulatePoints))
    # TODO: name the resulted 3D points as "points_3d"
    points_3d = real3D
    # points_3d = cv2.projectPoints(real3D, projector_R, projector_t, camera_K, camera_d)

    mask = (points_3d[:,:,2] > 200)&(points_3d[:,:,2] < 1400)
    final_3d =[]
    final_RGB = []
    for ma,i in enumerate(mask):
        if i[0]:
            final_3d.append([points_3d[ma][0].tolist()])
            final_RGB.append(RGB_value[ma])

    points_3d= np.asarray(final_3d)
    final_RGB = np.asarray(final_RGB)
    # return points_3d
    return points_3d,final_RGB
	
def write_3d_points(points_3d):
	
    # ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))

def write_3d_points_color(points_3d,RGB_value):

    # ===== DO NOT CHANGE THIS FUNCTION =====
    # print("write output point cloud")
    # print(points_3d.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name, "w") as f:
        for i,p in enumerate(points_3d):
            f.write("%d %d %d %d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2],RGB_value[i][0],RGB_value[i][1],RGB_value[i][2]))

    # return points_3d, camera_points, projector_points
    
if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====
	
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d,RGB_value = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
    write_3d_points_color(points_3d,RGB_value)
	

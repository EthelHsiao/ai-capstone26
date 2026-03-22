import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """

        ### TODO ###

        # =============================================
        # STEP 1: Build intrinsic matrix K
        # =============================================
        # Both cameras share the same intrinsics:
        #   512x512 resolution, 90° FOV, principal point at center
        W = self.width   # 512
        H = self.height  # 512
        fov_rad = np.radians(fov)
        f = (W / 2.0) / np.tan(fov_rad / 2.0)  # = 256
        cx = W / 2.0  # 256
        cy = H / 2.0  # 256

        K = np.array([
            [f,  0, cx],
            [0,  f, cy],
            [0,  0,  1]
        ])
        K_inv = np.linalg.inv(K)

        # =============================================
        # STEP 2: Build extrinsic parameters
        # =============================================
        # Habitat uses OpenGL convention:
        #   x = right, y = up, z = toward viewer
        #   Camera default looks along -z
        #
        # The pinhole matrix K uses OpenCV convention:
        #   x = right, y = down, z = forward
        #
        # We need gl2cv to convert between them (flip y and z):
        # gl2cv = np.diag([1.0, -1.0, -1.0])

        # BEV camera: position (0, 2.5, 0), orientation (-pi/2, 0, 0)
        # Rotation around x-axis by theta (pitch)
        theta_rad = np.radians(theta)  # theta = -90 -> -pi/2
        R_bev = np.array([
            [1,               0,                0],
            [0,  np.cos(theta_rad), -np.sin(theta_rad)],
            [0,  np.sin(theta_rad),  np.cos(theta_rad)]
        ])
        R_bev_inv = np.array([
            [1,               0,                0],
            [0,  np.cos(-theta_rad), -np.sin(-theta_rad)],
            [0,  np.sin(-theta_rad),  np.cos(-theta_rad)]
        ])
        t_bev_world = np.array([0, -2.5, 0]) # in opencv type
        t_bev_world_inv = np.array([0, 2.5, 0])

        # Front camera: position (0, 1, 0), orientation (0, 0, 0)
        R_front = np.eye(3)
        t_front_world = np.array([0, -1.0, 0]) # in opencv type

        # =============================================
        # STEP 3-6: Project each BEV pixel to front pixel
        # =============================================
        new_pixels = []

        for (u_bev, v_bev) in points:
            # -----------------------------------------
            # STEP 3: BEV pixel -> ray in BEV camera frame
            # -----------------------------------------
            # K_inv gives ray in OpenCV convention
            # Convert to OpenGL before using R_bev
            pixel_h = np.array([u_bev, v_bev, 1.0])
            ray_origin_bev = K_inv @ pixel_h
            # ray_gl = gl2cv @ ray_cv  # OpenCV -> OpenGL

            # -----------------------------------------
            # STEP 4: Ray -> 3D world point on ground (y=0)
            # -----------------------------------------
            # R_bev defines the camera frame orientation:
            #   R_bev @ direction_cam = direction_world
            ray_world = R_bev @ ray_origin_bev 

            # Parametric ray: P = origin + t * ray_world
            # Solve for ground plane y = 0:
            origin = t_bev_world
            t_param = -origin[1] / ray_world[1]
            point_world = origin + t_param * ray_world

            # -----------------------------------------
            # STEP 5: World point -> front camera frame
            # -----------------------------------------
            # P_cam_gl = R^T @ (P_world - t_cam)
            point_front= R_front.T @ (point_world - t_front_world)

            # -----------------------------------------
            # STEP 6: Project to front image pixel
            # -----------------------------------------
            # Convert OpenGL -> OpenCV, then apply K
            
            projected = K @ point_front
            print("point front: \n",point_front)
            print("projection: \n",projected)
            u_front = projected[0] / projected[2]
            v_front = projected[1] / projected[2]

            new_pixels.append([int(round(u_front)), int(round(v_front))])
            print(int(round(u_front)), " ",int(round(v_front)))

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    # front_rgb = "bev_data/front1.png"
    # top_rgb = "bev_data/bev1.png"

    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
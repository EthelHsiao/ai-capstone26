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

        ## 1. intrinsic matrix K for two cameras ( the same )
        f = (self.width / 2.0) / np.tan(np.radians(fov) / 2.0)  # focal / np.tan() takes radians
        cx = self.width / 2.0  # 256
        cy = self.height  / 2.0  # 256

        # intrinsic matrix K
        K = np.array([ 
            [f,  0, cx],
            [0,  f, cy],
            [0,  0,  1]
        ])
        # inverse K for pixel to ray_bev
        K_inv = np.linalg.inv(K)

        ## 2. extrinsic params

        # 2.1 BEV camera: position (0, 2.5, 0), orientation (-pi/2, 0, 0)
        theta_rad = np.radians(theta)
        # Rotation around x-axis by theta
        R_bev = np.array([
            [1,               0,                0],
            [0,  np.cos(theta_rad), -np.sin(theta_rad)],
            [0,  np.sin(theta_rad),  np.cos(theta_rad)]
        ])
        # Translation
        t_bev_world = np.array([0, 2.5, 0])

        # 2.2 Front camera: position (0, 1, 0), orientation (0, 0, 0)
        # Rotation (no rotation, facing front)
        R_front = np.eye(3) 
        # Translation
        t_front_world = np.array([0, 1.0, 0])

        ### 3. Convert BEV pixel to Front pixel
        new_pixels = []
        # gl2cv : convert GL to CV (flip y, z)
        gl2cv = np.diag([1.0, -1.0, -1.0])  # pinhole K uses openCV convention, but camera position and rotation uses openGL one (frlp y, z)

        for (u_bev, v_bev) in points:

            ## 3. BEV pixel -> ray in BEV camera
            # K_inv gives ray in OpenCV convention
            pixel = np.array([u_bev, v_bev, 1.0])  # add 1.0 for addition in multiplication
            ray_bev_cv = K_inv @ pixel
            ray_bev_gl = gl2cv @ ray_bev_cv  # OpenCV -> OpenGL


            ## 4. ray -> 3D world point on ground (y=0)
            #   ray_bev to ray_world
            ray_world = R_bev @ ray_bev_gl

            # Parametric ray: P = origin + t * ray_world
            # Solve for ground plane y = 0:
            origin = t_bev_world
            t = -origin[1] / ray_world[1]  #  y = 0, origin.y + t * ray_world.y = 0
            point_world = origin + t * ray_world

            
            ## 5. World point -> front camera
            # P_cam_gl = R^T @ (P_world - t_cam)
            point_front_gl = R_front.T @ (point_world - t_front_world)  # Rotation matrix is orthogonal -> inverse = transpose

            
            ## 6. Project to front pixel
            # Convert OpenGL -> OpenCV ( for applying K )
            point_front_cv = gl2cv @ point_front_gl
            projected = K @ point_front_cv
            # perspective division
            u_front = projected[0] / projected[2]
            v_front = projected[1] / projected[2]

            # append to new_pixels
            new_pixels.append([int(round(u_front)), int(round(v_front))])

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
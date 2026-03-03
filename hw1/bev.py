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
        self.points = points

    def top_to_front(self, theta=-90, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
        Project selected pixels from BEV (bird's-eye view) image to the front-view image.

        Assumptions:
            - BEV camera height = 2.5 m
            - Ground plane y = 0
        Parameters:
            theta, phi, gamma : rotation angles (degrees) around X, Y, Z axes
            dx, dy, dz        : translation offsets (meters)
            fov               : field of view (degrees)
        Returns:
            new_pixels : numpy array of projected pixel coordinates in the front view
        """

        # Ensure the input points exist and are float64
        points = np.array(self.points, dtype=np.float64)

        # ---------- Camera Intrinsics ----------
        focal = 256.0
        cx = self.width / 2.0
        cy = self.height / 2.0

        """
        {-focal} is because the direction of image(u, v) is (right, down)
        while camera (x, y, z) is (left, up, front)
        They are not consistent.
        """
        # Intrinsic matrix K
        K = np.array([
            [-focal, 0, cx],
            [0, -focal, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # ---------- BEV Camera Extrinsics ----------
        Y_cam = 2.5  # BEV camera height (meters)
        C_bev = np.array([0, Y_cam, 0], dtype=np.float64)  # Camera center in world coordinates

        # Rotation around X-axis by -90°: from top-down to front view
        theta = -90  # Override input to ensure top → front
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
            [0, np.sin(np.radians(theta)),  np.cos(np.radians(theta))]
        ], dtype=np.float64)

        R_bev = Rx

        # ---------- Front Camera Extrinsics ----------
        # Assume the front camera is at height 1 m, facing along the Z-axis
        C_front = np.array([0, 1, 0], dtype=np.float64)
        R_front = np.eye(3)

        # ---------- Projection from BEV to Front ----------
        new_pixels = []

        for (u, v) in points:
            # Step 1: Convert pixel to BEV camera coordinates (assuming depth = Y_cam)
            uv1 = np.array([u, v, 1.0], dtype=np.float64)
            P_cam_bev = Y_cam * (np.linalg.inv(K) @ uv1)

            # Step 2: Transform BEV camera coordinates to world coordinates
            P_world = R_bev.T @ P_cam_bev + C_bev

            # Step 3: Transform world coordinates to front camera coordinates
            P_front_cam = R_front @ (P_world - C_front)

            # Step 4: Project front camera coordinates onto image plane
            p_h = K @ P_front_cam

            # Step 5: Normalize homogeneous coordinates
            if p_h[2] <= 0:  # Point behind camera
                continue
            u_f = int(round(p_h[0] / p_h[2]))
            v_f = int(round(p_h[1] / p_h[2]))

            new_pixels.append([u_f, v_f])

        # Return result as numpy array
        if len(new_pixels) == 0:
            return np.zeros((0, 2), dtype=np.int32)

        return np.array(new_pixels, dtype=np.int32)



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


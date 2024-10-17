import cv2
import numpy as np
import os
import plotly.graph_objects as go
from tqdm import tqdm
from inference.imageutils import Image


class processLogImage:
    def __init__(self):
        self.Image_ = Image()

    def generate3DHistogram(
        self,
        image,
        color_image=None,
        save_name="pcl.html",
        title="pcl",
        vector=None,
        plane=None,
    ):
        upper_lim = np.max(image)
        lower_lim = np.min(image)

        r, g, b = image[:, 0], image[:, 1], image[:, 2]
        r = r.flatten()
        g = g.flatten()
        b = b.flatten()

        if color_image is None:
            color_image = image

        r_color, g_color, b_color = (
            color_image[:, 0],
            color_image[:, 1],
            color_image[:, 2],
        )

        r_color = r_color.flatten()
        g_color = g_color.flatten()
        b_color = b_color.flatten()

        colors = []
        for red, blue, green in zip(r_color, g_color, b_color):
            colors.append((red, blue, green))
        x_vals, y_vals, z_vals = r, g, b
        sizes = 1

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode="markers",
                    marker=dict(size=sizes, color=colors, opacity=1),
                )
            ]
        )

        if vector is not None:
            for vec in vector:
                vector_start = vec[0]
                vector_end = vec[1]
                fig.add_trace(
                    go.Scatter3d(
                        x=[vector_start[0], vector_end[0]],
                        y=[vector_start[1], vector_end[1]],
                        z=[vector_start[2], vector_end[2]],
                        mode="lines",
                        line=dict(color="red", width=5),
                        name="Vector",
                    )
                )

        if plane is not None:
            for pl in plane:
                a, b, c, d = pl
                xx, yy = np.meshgrid(np.linspace(0, 3, 10), np.linspace(0, 3, 10))
                zz = (-a * xx - b * yy - d) / c
                fig.add_trace(
                    go.Surface(
                        x=xx,
                        y=yy,
                        z=zz,
                        colorscale="Viridis",
                        opacity=0.7,
                        name="Plane",
                    )
                )

        fig.update_layout(
            scene=dict(
                xaxis_title="Red",
                yaxis_title="Green",
                zaxis_title="Blue",
            ),
            title=title,
        )

        fig.write_html(save_name)
        print(f"PCL saved at {save_name}")

    def estimateTransformationMat(self, normal_vector):
        print("Vector from Shadow to Lit:", normal_vector)
        # Normalize the vector to get the normal of the plane
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        # Find two orthogonal vectors on the plane (using Gram-Schmidt process)
        # We can choose an arbitrary vector that is not parallel to the normal
        arbitrary_vector = (
            np.array([1, 0, 0])
            if not np.allclose(normal_vector, [1, 0, 0])
            else np.array([0, 1, 0])
        )

        # Use the Gram-Schmidt process to find a vector orthogonal to the normal
        v1 = arbitrary_vector - np.dot(arbitrary_vector, normal_vector) * normal_vector
        v1 = v1 / np.linalg.norm(v1)  # Normalize v1

        # Find another orthogonal vector to both normal_vector and v1 using cross product
        v2 = np.cross(normal_vector, v1)
        v2 = v2 / np.linalg.norm(v2)  # Normalize v2

        # Create the transformation matrix (3D to 2D projection onto the plane)
        transformation_matrix = np.vstack([v1, v2])  # Stack v1 and v2 into a 2x3 matrix
        return transformation_matrix

    def transformPoints(self, image, tf_mat):
        image = image.reshape(-1, 3)
        projected_points = np.dot(image, tf_mat.T)
        return projected_points

    def estimateGrayscaleImagefromXY(self, projected_points, image_shape):
        x_coords = projected_points[:, 0]
        y_coords = projected_points[:, 1]

        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * 255
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * 255

        grayscale_values = 0.5 * (x_norm + y_norm)
        grayscale_image = grayscale_values.astype(np.uint8)

        grayscale_image = grayscale_image.reshape(image_shape[0], image_shape[1])

        return grayscale_image

    def findExtremePoints(self, points):
        if points.shape[0] < points.shape[1]:
            raise Exception(
                f"shape of points should be of form num_points,dim, got {points.shape}"
            )
        max_dist = 0
        farthest_points = None
        num_points = len(points)

        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_dist:
                    max_dist = dist
                    farthest_points = (points[i], points[j])

        return farthest_points, max_dist

    def filterPointsbyNearestNeighbor(self, points, threshold=1.0, min_neighbors=5):
        valid_points = []

        num_points = len(points)
        for i in range(num_points):
            distances = np.linalg.norm(points - points[i], axis=1)

            # Count how many points are within the threshold (excluding the point itself)
            neighbor_count = (
                np.sum(distances < threshold) - 1
            )  # Subtract 1 to exclude the point itself

            # If the point has at least min_neighbors, add it to the valid points list
            if neighbor_count >= min_neighbors:
                valid_points.append(points[i])

        valid_points = np.array(valid_points)
        return valid_points

    def projectPoint(self, point, plane_normal, D):
        point = np.array(point)
        plane_normal = np.array(plane_normal)

        # Step 1: Compute the 3D projected point onto the plane (same as before)
        distance = (np.dot(plane_normal, point) + D) / np.dot(
            plane_normal, plane_normal
        )
        projected_point_3D = point - distance * plane_normal

        # Step 2: Find two vectors (basis) that lie on the plane (orthogonal to the normal)
        # We can use the cross product with any arbitrary vector that is not parallel to the normal.
        arbitrary_vector = (
            np.array([1, 0, 0])
            if not np.allclose(plane_normal, [1, 0, 0])
            else np.array([0, 1, 0])
        )
        basis1 = np.cross(plane_normal, arbitrary_vector)
        basis1 /= np.linalg.norm(basis1)  # Normalize the vector

        basis2 = np.cross(plane_normal, basis1)  # The second basis vector
        basis2 /= np.linalg.norm(basis2)  # Normalize the vector

        # Step 3: Compute the 2D coordinates by projecting the 3D point onto the basis vectors
        local_2D_x = np.dot(projected_point_3D, basis1)  # Project onto basis1
        local_2D_y = np.dot(projected_point_3D, basis2)  # Project onto basis2

        return np.array([local_2D_x, local_2D_y])

    def reprojectXYtoRGB(self, pts_2d, constant_intensity=150):
        rgb_values = []
        x_vals, y_vals = pts_2d[:, 0], pts_2d[:, 1]
        for x, y in zip(x_vals, y_vals):
            x_normalized = (
                (x - np.min(x_vals)) / (np.max(x_vals) - np.min(x_vals)) * 255
            )
            y_normalized = (
                (y - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals)) * 255
            )

            R = x_normalized
            G = y_normalized

            # Ensure that R + G + B = constant_intensity
            B = constant_intensity - (R + G)

            # Clip RGB values to the range [0, constant_intensity] to avoid negatives
            R = np.clip(R, 0, constant_intensity)
            G = np.clip(G, 0, constant_intensity)
            B = np.clip(B, 0, constant_intensity)

            rgb_values.append([R, G, B])

        return np.array(rgb_values)

    def reprojectXYtoOriginalColor(self, pts_2d, point_color_dict, intensity=None):
        rgb_values = []
        x_vals, y_vals = pts_2d[:, 0], pts_2d[:, 1]
        min_x, min_y = np.min(x_vals), np.min(y_vals)
        max_x, max_y = np.max(x_vals), np.max(y_vals)

        for x, y in tqdm(zip(x_vals, y_vals), desc="Reprojecting plane points to RGB"):
            x = int(((x - min_x) / (max_x - min_x)) * 255)
            y = int(((y - min_y) / (max_y - min_y)) * 255)

            pt = tuple((x, y))

            rgb_arr = np.array(list(point_color_dict[pt]))
            rgb_intensity = np.sum(rgb_arr, axis=1)
            sorted_indices = np.argsort(rgb_intensity)

            max_idx = sorted_indices[-1]
            R, G, B = rgb_arr[max_idx, :]

            rgb_values.append([R, G, B])

        rgb_values = np.array(rgb_values)

        if intensity is not None:
            intensity_rgb_values = np.mean(rgb_values, axis=1)

            scale_factor_rgb_values = np.divide(
                intensity, intensity_rgb_values, where=intensity_rgb_values != 0
            )
            rgb_values = rgb_values * scale_factor_rgb_values[:, np.newaxis]
            rgb_values = rgb_values.astype(np.uint8)
        return rgb_values

    def calculate_kernel_average(self, image, row, col, kernel_size):
        kernel_size = kernel_size[0]
        half_kernel = kernel_size // 2

        height, width, channels = image.shape

        # Check if the kernel is within the image bounds, pad if necessary
        row_start = max(0, row - half_kernel)
        row_end = min(height, row + half_kernel + 1)
        col_start = max(0, col - half_kernel)
        col_end = min(width, col + half_kernel + 1)

        # Extract the 3x3 patch from the image
        patch = image[row_start:row_end, col_start:col_end, :]

        # Calculate the  value across the 3x3 patch for each channel
        kernel_average = np.mean(patch, axis=(0, 1))

        return kernel_average

    def estimateISDwithNeigbors(self, logimage, lit_shadow_pts, kernel=(3, 3)):
        lit_pt = lit_shadow_pts[0]
        shadow_pt = lit_shadow_pts[1]
        lit_pt_logval = self.calculate_kernel_average(
            logimage, lit_pt[0], lit_pt[1], kernel
        )
        shadow_pt_logval = self.calculate_kernel_average(
            logimage, shadow_pt[0], shadow_pt[1], kernel
        )
        ISD_vec = np.array(lit_pt_logval - shadow_pt_logval)
        return ISD_vec

    def fitPlane(self, points, sufficient_inliers, max_iterations=1000, threshold=1e-2):
        best_plane = None
        best_inliers_count = 0
        best_inliers_mask = None

        num_points = points.shape[0]

        # RANSAC iterations
        for _ in tqdm(range(max_iterations), desc="Estimating best fit plane for RGB"):
            # Randomly select 3 points
            sample_indices = np.random.choice(num_points, 3, replace=False)
            sample_points = points[sample_indices]

            # Compute the normal of the plane from the 3 points
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)

            # Ensure the normal is not degenerate (i.e., points are not collinear)
            if np.linalg.norm(normal) == 0:
                continue

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)

            # Plane equation: ax + by + cz + d = 0
            # We can use any of the sample points to solve for d
            a, b, c = normal
            d = -np.dot(normal, sample_points[0])

            # Compute the distance from all points to the plane
            distances = np.abs(
                a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
            )

            # Find the inliers: points where the distance is less than the threshold
            inliers_mask = distances < threshold
            inliers_count = np.sum(inliers_mask)

            # Update the best model if the current one has more inliers
            if inliers_count > best_inliers_count:
                best_plane = (a, b, c, d)
                best_inliers_count = inliers_count
                best_inliers_mask = inliers_mask
                if best_inliers_count >= sufficient_inliers:
                    break

        return best_plane, best_inliers_mask

    def getNormaltoPlane(self, original_plane):
        a, b, c, _ = original_plane

        # Arbitrarily choose two components for the new normal vector
        a_prime = 1
        b_prime = 0

        # Solve for c' using the orthogonality condition a * a' + b * b' + c * c' = 0
        c_prime = -(a * a_prime + b * b_prime) / c if c != 0 else 0

        # Define the new perpendicular plane equation: a'x + b'y + c'z + d' = 0
        # d' can be arbitrary
        d_prime = 0
        return np.array([a_prime, b_prime, c_prime, d_prime])

    def fitLine(self, points, max_iterations=1000, threshold=1e-2):
        best_line = None
        best_inliers_count = 0
        best_inliers_mask = None

        intensity_points = np.mean(points, axis=1)
        intensity_points_argsorted = np.argsort(intensity_points)

        bright_pts = intensity_points_argsorted[0 : int(len(intensity_points) * 0.05)]
        dim_pts = intensity_points_argsorted[
            int(len(intensity_points) * 0.95) : len(intensity_points)
        ]

        num_points = points.shape[0]

        for _ in tqdm(range(max_iterations), desc="Estimating best line using RANSAC"):
            # Randomly select 2 points to define the line
            # sample_indices = np.random.choice(num_points, 2, replace=False)
            # p1, p2 = points[sample_indices]

            p1 = points[np.random.choice(dim_pts)]
            p2 = points[np.random.choice(bright_pts)]

            # Compute the direction vector of the line
            direction_vector = p2 - p1
            direction_vector = direction_vector / np.linalg.norm(
                direction_vector
            )  # Normalize

            # Define the line in parametric form: P(t) = P0 + t * d
            p0 = p1  # Use the first point as P0

            # Compute the distance of all points to the line
            distances = np.linalg.norm(np.cross(points - p0, direction_vector), axis=1)
            distances /= np.linalg.norm(direction_vector)  # Normalize distance

            # Identify inliers (points within the threshold distance to the line)
            inliers_mask = distances < threshold
            inliers_count = np.sum(inliers_mask)

            # Update the best line if the current one has more inliers
            if inliers_count > best_inliers_count:
                best_line = (p0, direction_vector)
                best_inliers_count = inliers_count
                best_inliers_mask = inliers_mask

        return [p0, p0 + 10 * direction_vector], best_inliers_mask

    def getGrayImagebyPointSelection(self, log_img, rgb_img, num_points=2, title="img"):
        while True:
            points = self.Image_.getPoints(rgb_img, num_points=num_points, title=title)
            lit_shadow_pts = [points[-2], points[-1]]  # lit, shadow
            ISD_vec = self.estimateISDwithNeigbors(
                log_img, lit_shadow_pts, kernel=(5, 5)
            )
            lit_shadow_pts = [
                log_img[points[-2][0], points[-2][1], :],
                log_img[points[-1][0], points[-1][1], :],
            ]

            ISD_vec = lit_shadow_pts[0] - lit_shadow_pts[1]

            tfmat = self.estimateTransformationMat(ISD_vec)
            projected_points_2d = self.transformPoints(log_img, tfmat)
            gray_img = self.estimateGrayscaleImagefromXY(
                projected_points_2d, log_img.shape
            )
            self.Image_.showImage(gray_img)
            print("Proceed with image or Re-do? Choose '1', '2'")
            print("1: Proceed")
            print("2: Re-Do")
            choice = input()
            if choice == "1":
                break
        return gray_img

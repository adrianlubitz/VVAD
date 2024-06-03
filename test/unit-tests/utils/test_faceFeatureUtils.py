import unittest
import numpy as np

import wildvvad.utils.faceFeatureUtils as fu

class TestFeatureUtils(unittest.TestCase):
    def test_translate_point_cloud(self):
        """
            Unit test on the linear translation of a 3 dimensional point cloud.
            All entities of that point cloud should be tranlated by the translation
            vector.
        """

        point_cloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        translation_vector = np.array([1, 1, 1])
        expected_result = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        result = fu.translate_point_cloud(point_cloud, translation_vector)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_rotate_point_cloud_matrix(self):
        """
            Unit test on the rotation of a given 3 dimensional point cloud matrix.
            The rotation is around a three-dimensional rotation matrix.
        """

        point_cloud = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotation_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        expected_result = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        result = fu.rotate_point_cloud_matrix(point_cloud, rotation_matrix)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_compute_centroid(self):
        """
            Unit test of calculation of the centroid of a 3 dimensional point cloud
            matrix.
        """

        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_result = np.array([4, 5, 6])
        result = fu.compute_centroid(points)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_compute_rotation_angle(self):
        """
            Unit test on the computation of the rotation angle of a line (formed by
            multiple points) on the x-y-plane
        """

        nose_bridge_landmarks = np.array([[0, 0], [1, 1]])
        expected_result = np.pi / 4  # 45 degrees in radians
        result = fu.compute_rotation_angle(nose_bridge_landmarks)
        self.assertAlmostEqual(result, expected_result)

    def test_rotate_point_cloud_angle(self):
        """
            Unit test on the rotation of a 3 dimensional point cloud around an angle
            in the x-y-plane.
        """

        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        angle = np.pi / 2  # 90 degrees in radians
        expected_result = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        result = fu.rotate_point_cloud_angle(points, angle)
        self.assertTrue(np.array_equal(np.round(result), expected_result))


if __name__ == '__main__':
    unittest.main()

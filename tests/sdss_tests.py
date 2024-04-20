import unittest
import pandas as pd
import numpy as np
from src.data_models.sdss import SDSS


class TestSDSS(unittest.TestCase):
    """
    Unit test class for the SDSS class.
    """

    def setUp(self) -> None:
        """
        Set up the test case.

        :return: None
        """
        data = pd.DataFrame({
            'ra': np.linspace(130, 230, 100),
            'de': np.linspace(5, 65, 100),
            'z_redshift': np.linspace(0.05, 0.15, 100),
            'u': np.linspace(5, 10, 100),
            'g': np.random.uniform(0, 1, 100),
            'r': np.linspace(0, 10, 100),
            'i': np.random.uniform(0, 1, 100),
            'z_magnitude': np.random.uniform(0, 1, 100)
        })
        self.sdss = SDSS(data)

    def test_sample_down(self) -> None:
        """
        Test the sample_down method.

        :return: None
        """
        self.assertRaises(ValueError, self.sdss.sample_down, n=101)
        self.sdss.sample_down(n=10)
        self.assertEqual(self.sdss.data.shape[0], 10)

    def test_select_redshift_slice(self) -> None:
        """
        Test the select_redshift_slice method.

        :return: None
        """
        self.sdss.select_redshift_slice(z_min=0.08, z_max=0.09)
        self.assertTrue(all(self.sdss.data['z_redshift'] >= 0.08))
        self.assertTrue(all(self.sdss.data['z_redshift'] <= 0.09))
        self.assertEqual(self.sdss.data.shape[0], 10)

    def test_coordinates(self) -> None:
        """
        Test the _standardize_coordinates method.

        :return: None
        """

        de = np.linspace(130, 230, 100)
        ra = np.linspace(5, 65, 100)

        de_ref = (130 - np.mean(de)) / np.std(de)
        ra_ref = (5 - np.mean(ra)) / np.std(ra)

        self.sdss._standardize_coordinates()
        self.assertAlmostEqual(self.sdss._de_standardized[0], ra_ref, 6)
        self.assertAlmostEqual(self.sdss._ra_standardized[0], de_ref, 6)



if __name__ == '__main__':
    unittest.main()

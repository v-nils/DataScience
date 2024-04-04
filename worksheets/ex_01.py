import pandas as pd
from src.data_models import SdssGalaxy


if __name__ == "__main__":
    sdss_galaxy = SdssGalaxy(pd.read_csv('data/raw_data/sdss_cutout.csv'))

    print(sdss_galaxy.data.head())
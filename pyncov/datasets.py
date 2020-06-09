# -*- coding: utf-8 -*-
from warnings import warn
from pyncov.io import _unverified_stream_url
from io import StringIO
import pkgutil

try:
    import pandas as pd
except ImportError:
    warn(
        """The pyncov.data module requires Pandas to parse the data.
    You can install all the dependencies with pip install pyncov[all]
    """
    )
    raise ImportError(
        "pyncov.data requires Pandas to parse and process data"
    ) from None


# Data provided by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University
URL_FATALITIES_CSSE_GLOBAL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master' \
                             '/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

URL_FATALITIES_CSSE_USA = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                          '/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'


def load_csse_global_fatalities(url=URL_FATALITIES_CSSE_GLOBAL, verify=False):
    if verify:
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(_unverified_stream_url(url))
    df = df.rename(columns={
        'Province/State': 'n2',
        'Country/Region': 'n1'
    }).drop(['Lat', 'Long'], axis=1)
    df_iso_codes = pd.read_csv(StringIO(pkgutil.get_data('pyncov.data', 'iso-codes.csv').decode('UTF-8')))
    df_population = pd.read_csv(StringIO(pkgutil.get_data('pyncov.data', 'population.csv').decode('UTF-8')))
    df = df.merge(df_iso_codes, on='n1').merge(df_population, on='iso3').set_index(['iso3', 'iso2', 'n1', 'n2', 'population'])
    df.columns = pd.DatetimeIndex(df.columns, freq='D')
    return df


def load_csse_usa_fatalities(url=URL_FATALITIES_CSSE_USA, verify=False):
    if verify:
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(_unverified_stream_url(url))
    df = df.rename(
        columns={'Country_Region': 'n2',
                 'Province_State': 'n1',
                 'Population': 'population'
                 }
    ).drop([
        'UID', 'FIPS', 'Admin2', 'Lat',
        'Long_', 'Combined_Key', 'code3'], axis=1
    )
    df = df.set_index(['iso3', 'iso2', 'n1', 'n2', 'population'])
    df.columns = pd.DatetimeIndex(df.columns, freq='D')
    return df


def load_csse_world_fatalities(verify=False):
    return pd.concat([load_csse_usa_fatalities(verify=verify), load_csse_global_fatalities(verify=verify)])

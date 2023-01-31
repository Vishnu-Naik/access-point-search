#!/usr/bin/env python
# coding: utf-8



from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from pathlib import Path
import os

# get_ipython().run_line_magic('matplotlib', 'inline')

from statsmodels.tsa.stattools import (
    adfuller,
    kpss)

CUR_DIR = CUR_DIR = Path(__file__).parent.parent.resolve()
DATA_CSV_FILE_NAME = '../data/data_collected_v1.csv'


class StationaryTester:
    def __init__(self, adf_autolag: str = 'AIC'):
        """
        A helper class using which two well known stationary check called `ADF` and `KPSS` test can be done on any series.
        If the series has any known trends then it is recommended to employ KPSS test as well. In order to do so just
        pass has_trends parameter as 'True'.

        Args:
        adf_autolag: takes following values:
            {‘AIC’, ‘BIC’, ‘t-stat’, None}, optional
            - If None, then maxlag = 12*(nobs/100)^{1/4} is used where nobs is the number of observations.
            - If ‘AIC’, then adfuller will choose the number of lags that yields the lowest AIC
            - If p-value is close to significant, increase the number of lags.
        has_trends: boolean value for trends

        For example:

        stat_tester = StationaryTester()
        stat_tester.test(data_frame['signal1'].to_numpy())
        """

        self.adf_autolag = adf_autolag

    @staticmethod
    def adf_test(series, adf_autolag, verbose=False):
        #
        # And if the p-value is still greater than significance level of 0.05 and the ADF statistic is
        # higher than any of the critical values. Clearly, there is no reason to reject the null hypothesis.
        # So, the time series is in fact non-stationary. As shown in below example.

        statistic, p_value, n_lags, _, critical_values, _ = adfuller(series, autolag=adf_autolag)
        # Format Output
        if verbose:
            print('*' * 50)
            print(f'ADF Statistic: {statistic}')
            print('*' * 50)
            print(f'p-value: {p_value}')
            print(f'num lags: {n_lags}')
            print('Critial Values:')
            for key, value in critical_values.items():
                print(f'\t{key} : {value}')
            print('_' * 50)
            # The p-value is very less than the significance level of 0.05 and
            # hence we can reject the null hypothesis and take that the series is stationary.
            print(f'Result: The series is {"not " if p_value > 0.05 else ""}stationary')
            print('_' * 50)
        return p_value < 0.05

    @staticmethod
    def kpss_test(series, verbose=False, **kw):
        """The KPSS test, short for, Kwiatkowski-Phillips-Schmidt-Shin (KPSS),
        is a type of Unit root test that tests for the stationarity of a given
        series around a deterministic trend."""

        statistic, p_value, n_lags, critical_values = kpss(series, **kw)
        if verbose:
            print('*' * 50)
            print(f'KPSS Statistic: {statistic}')
            print('*' * 50)
            print(f'p-value: {p_value}')
            print(f'num lags: {n_lags}')
            print('Critial Values:')
            for key, value in critical_values.items():
                print(f'\t{key} : {value}')
            print('_' * 50)
            print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
            print('_'*50)

        return p_value > 0.05

    def test(self, series, has_trends=False, verbose=False):
        adf_result = self.adf_test(series, self.adf_autolag, verbose=verbose)
        kpss_result = True
        if has_trends:
            kpss_result = self.kpss_test(series, regression='ct', verbose=verbose)
        return adf_result and kpss_result


def main():
    # ADF test
    # ## Sine signal
    sine_wave_df = pd.read_csv(CUR_DIR/'data/data_collected_v1.csv', names=['signal1'])
    x_signal = sine_wave_df['signal1'].values
    StationaryTester.adf_test(x_signal, adf_autolag='AIC', verbose=True)
    # ## Exo-leg signal
    exo_leg_data_df = pd.read_csv(CUR_DIR/'data/exo_hip_right_2022_12_13-18_02.csv')
    StationaryTester.adf_test(exo_leg_data_df.values, adf_autolag='AIC')

    ####################################################################################################################
    # KPSS test
    # ### Non stationary series with deterministic trends
    path = 'https://raw.githubusercontent.com/selva86/datasets/master/livestock.csv'
    df = pd.read_csv(path, parse_dates=['date'], index_col='date')
    df.plot(title='Annual Livestock', figsize=(12, 7), legend=None);

    # KPSS test: Stationary test around mean
    non_stat_series_with_determinitic_trend = df.loc[:, 'value'].values
    StationaryTester.kpss_test(non_stat_series_with_determinitic_trend)

    # Stationary test around deterministic trend
    StationaryTester.kpss_test(non_stat_series_with_determinitic_trend, regression='ct')

    # # Testing exo-leg signal
    # ## Exo-leg signal
    StationaryTester.kpss_test(exo_leg_data_df, regression='ct')




if __name__ == '__main__':
    main()


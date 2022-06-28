{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "i_ZjFcizpOlV",
    "outputId": "72add5ef-3109-4a08-ceb0-3c0253f544d9"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from statsmodels.tsa.stattools import adfuller\n",
    "import pandas as pd\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt\n",
    "from fbprophet import Prophet\n",
    "mpl.rcParams['figure.figsize'] = (10, 8)\n",
    "mpl.rcParams['axes.grid'] = False\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "#from sktime.forecasting.model_selection import temporal_train_test_split\n",
    "#from sktime.utils.plotting.forecasting import plot_ys\n",
    "%matplotlib inline\n",
    "\n",
    "from sktime.forecasting.base import ForecastingHorizon\n",
    "#from sktime.forecasting.model_selection import temporal_train_test_split\n",
    "from sktime.performance_metrics.forecasting import mean_absolute_percentage_error\n",
    "from sktime.utils.plotting import plot_series"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "UZzq6IP93BlA"
   },
   "outputs": [],
   "source": [
    "df=pd.read_csv('/Users/kasturid3/Desktop/vaccines_by_age.csv',parse_dates=['Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Agegroup</th>\n",
       "      <th>At least one dose_cumulative</th>\n",
       "      <th>Second_dose_cumulative</th>\n",
       "      <th>fully_vaccinated_cumulative</th>\n",
       "      <th>third_dose_cumulative</th>\n",
       "      <th>Total population</th>\n",
       "      <th>Percent_at_least_one_dose</th>\n",
       "      <th>Percent_fully_vaccinated</th>\n",
       "      <th>Percent_3doses</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2020-12-16</td>\n",
       "      <td>12-17yrs</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>951519</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2020-12-16</td>\n",
       "      <td>18-29yrs</td>\n",
       "      <td>45</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2455535</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2020-12-16</td>\n",
       "      <td>30-39yrs</td>\n",
       "      <td>66</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2056059</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2020-12-16</td>\n",
       "      <td>40-49yrs</td>\n",
       "      <td>98</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1876583</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2020-12-16</td>\n",
       "      <td>50-59yrs</td>\n",
       "      <td>141</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2060934</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5418</th>\n",
       "      <td>2022-03-31</td>\n",
       "      <td>80</td>\n",
       "      <td>675107</td>\n",
       "      <td>NaN</td>\n",
       "      <td>659945.0</td>\n",
       "      <td>580715.0</td>\n",
       "      <td>655835</td>\n",
       "      <td>0.9999</td>\n",
       "      <td>0.9999</td>\n",
       "      <td>0.8855</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5419</th>\n",
       "      <td>2022-03-31</td>\n",
       "      <td>Adults_18plus</td>\n",
       "      <td>11140351</td>\n",
       "      <td>NaN</td>\n",
       "      <td>10873399.0</td>\n",
       "      <td>7045917.0</td>\n",
       "      <td>11971129</td>\n",
       "      <td>0.9306</td>\n",
       "      <td>0.9083</td>\n",
       "      <td>0.5886</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5420</th>\n",
       "      <td>2022-03-31</td>\n",
       "      <td>Ontario_12plus</td>\n",
       "      <td>12004141</td>\n",
       "      <td>NaN</td>\n",
       "      <td>11759405.0</td>\n",
       "      <td>7179239.0</td>\n",
       "      <td>12932471</td>\n",
       "      <td>0.9282</td>\n",
       "      <td>0.9093</td>\n",
       "      <td>0.5551</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5421</th>\n",
       "      <td>2022-03-31</td>\n",
       "      <td>Ontario_5plus</td>\n",
       "      <td>12603706</td>\n",
       "      <td>NaN</td>\n",
       "      <td>12116728.0</td>\n",
       "      <td>7179322.0</td>\n",
       "      <td>14010998</td>\n",
       "      <td>0.8996</td>\n",
       "      <td>0.8648</td>\n",
       "      <td>0.5124</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5422</th>\n",
       "      <td>2022-03-31</td>\n",
       "      <td>Undisclosed_or_missing</td>\n",
       "      <td>2027</td>\n",
       "      <td>NaN</td>\n",
       "      <td>954.0</td>\n",
       "      <td>55.0</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5423 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           Date                Agegroup  At least one dose_cumulative  \\\n",
       "0    2020-12-16                12-17yrs                             0   \n",
       "1    2020-12-16                18-29yrs                            45   \n",
       "2    2020-12-16                30-39yrs                            66   \n",
       "3    2020-12-16                40-49yrs                            98   \n",
       "4    2020-12-16                50-59yrs                           141   \n",
       "...         ...                     ...                           ...   \n",
       "5418 2022-03-31                      80                        675107   \n",
       "5419 2022-03-31           Adults_18plus                      11140351   \n",
       "5420 2022-03-31          Ontario_12plus                      12004141   \n",
       "5421 2022-03-31           Ontario_5plus                      12603706   \n",
       "5422 2022-03-31  Undisclosed_or_missing                          2027   \n",
       "\n",
       "      Second_dose_cumulative  fully_vaccinated_cumulative  \\\n",
       "0                        0.0                          NaN   \n",
       "1                        0.0                          NaN   \n",
       "2                        0.0                          NaN   \n",
       "3                        0.0                          NaN   \n",
       "4                        0.0                          NaN   \n",
       "...                      ...                          ...   \n",
       "5418                     NaN                     659945.0   \n",
       "5419                     NaN                   10873399.0   \n",
       "5420                     NaN                   11759405.0   \n",
       "5421                     NaN                   12116728.0   \n",
       "5422                     NaN                        954.0   \n",
       "\n",
       "      third_dose_cumulative  Total population  Percent_at_least_one_dose  \\\n",
       "0                       NaN            951519                     0.0000   \n",
       "1                       NaN           2455535                     0.0000   \n",
       "2                       NaN           2056059                     0.0000   \n",
       "3                       NaN           1876583                     0.0000   \n",
       "4                       NaN           2060934                     0.0000   \n",
       "...                     ...               ...                        ...   \n",
       "5418               580715.0            655835                     0.9999   \n",
       "5419              7045917.0          11971129                     0.9306   \n",
       "5420              7179239.0          12932471                     0.9282   \n",
       "5421              7179322.0          14010998                     0.8996   \n",
       "5422                   55.0                 0                        NaN   \n",
       "\n",
       "      Percent_fully_vaccinated  Percent_3doses  \n",
       "0                       0.0000             NaN  \n",
       "1                       0.0000             NaN  \n",
       "2                       0.0000             NaN  \n",
       "3                       0.0000             NaN  \n",
       "4                       0.0000             NaN  \n",
       "...                        ...             ...  \n",
       "5418                    0.9999          0.8855  \n",
       "5419                    0.9083          0.5886  \n",
       "5420                    0.9093          0.5551  \n",
       "5421                    0.8648          0.5124  \n",
       "5422                       NaN             NaN  \n",
       "\n",
       "[5423 rows x 10 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "bjE94xvA3Boo"
   },
   "outputs": [],
   "source": [
    "df=df[['Date','Agegroup','At least one dose_cumulative']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "_-7RPpXa1s0d"
   },
   "outputs": [],
   "source": [
    "df=df[df['Date']>'2021-06-01'].reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "CNqFoZ1XyHZp"
   },
   "outputs": [],
   "source": [
    "df1=df.pivot_table(index=['Date'],columns='Agegroup',values=['At least one dose_cumulative']).reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "slPCcd3iyOlW"
   },
   "outputs": [],
   "source": [
    "df1.columns=['Date','05-11yrs','12-17yrs','18-29yrs','30-39yrs','40-49yrs','50-59yrs','60-69yrs','70-79yrs','80','Adults_18plus','Ontario_12plus','Ontario_5plus','Undisclosed_or_missing']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "tKxoaGduyX6p"
   },
   "outputs": [],
   "source": [
    "df2=df1[['Date','12-17yrs', '18-29yrs', '30-39yrs', '40-49yrs',\n",
    "       '50-59yrs', '60-69yrs', '70-79yrs', '80','Adults_18plus', 'Ontario_12plus']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 423
    },
    "id": "G5w-dvP2zrZ0",
    "outputId": "67891fb7-4cac-4ac8-98de-88bfb33c0b9c"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'df2' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-2-2068723f1203>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mdf2\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'df2' is not defined"
     ]
    }
   ],
   "source": [
    "df2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "df2=df2.set_index('Date')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "df2.index = pd.to_datetime(df2.index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "df2.index=df2.index.to_period(\"D\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = pd.Series(df2['12-17yrs'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "train=df2['12-17yrs'][:280]\n",
    "test=df2['12-17yrs'][280:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Date\n",
       "2022-03-09    861468.0\n",
       "2022-03-10    861617.0\n",
       "2022-03-11    861733.0\n",
       "2022-03-12    861879.0\n",
       "2022-03-13    862044.0\n",
       "2022-03-14    862098.0\n",
       "2022-03-15    862226.0\n",
       "2022-03-16    862355.0\n",
       "2022-03-17    862509.0\n",
       "2022-03-18    862660.0\n",
       "2022-03-19    862812.0\n",
       "2022-03-20    862926.0\n",
       "2022-03-21    862978.0\n",
       "2022-03-22    863055.0\n",
       "2022-03-23    863157.0\n",
       "2022-03-24    863223.0\n",
       "2022-03-25    863319.0\n",
       "2022-03-26    863427.0\n",
       "2022-03-27    863538.0\n",
       "2022-03-28    863576.0\n",
       "2022-03-29    863623.0\n",
       "2022-03-30    863704.0\n",
       "2022-03-31    863790.0\n",
       "Name: 12-17yrs, dtype: float64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.\n",
      "  warnings.warn('No frequency information was'\n",
      "/opt/anaconda3/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.\n",
      "  warnings.warn('No frequency information was'\n",
      "/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:547: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available\n",
      "  warnings.warn('Inverting hessian failed, no bse or cov_params '\n",
      "/opt/anaconda3/lib/python3.8/site-packages/statsmodels/tsa/arima_model.py:472: FutureWarning: \n",
      "statsmodels.tsa.arima_model.ARMA and statsmodels.tsa.arima_model.ARIMA have\n",
      "been deprecated in favor of statsmodels.tsa.arima.model.ARIMA (note the .\n",
      "between arima and model) and\n",
      "statsmodels.tsa.SARIMAX. These will be removed after the 0.12 release.\n",
      "\n",
      "statsmodels.tsa.arima.model.ARIMA makes use of the statespace framework and\n",
      "is both well tested and maintained.\n",
      "\n",
      "To silence this warning and continue using ARMA and ARIMA until they are\n",
      "removed, use:\n",
      "\n",
      "import warnings\n",
      "warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',\n",
      "                        FutureWarning)\n",
      "warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',\n",
      "                        FutureWarning)\n",
      "\n",
      "  warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "from statsmodels.tsa.arima_model import ARIMA\n",
    "\n",
    "# 1,1,2 ARIMA Model\n",
    "model = ARIMA(train, order=(1,1,2))\n",
    "model=model.fit()\n",
    "pred=model.predict(start=1, end=24, exog=None, dynamic=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2021-06-03      864.347909\n",
       "2021-06-04    18369.771746\n",
       "2021-06-05    19591.993838\n",
       "2021-06-06    19558.926732\n",
       "2021-06-07    19070.048390\n",
       "2021-06-08    17365.879850\n",
       "2021-06-09    16897.773946\n",
       "2021-06-10    17173.358767\n",
       "2021-06-11    16836.124034\n",
       "2021-06-12    16388.226721\n",
       "2021-06-13    14871.047048\n",
       "2021-06-14    14226.238294\n",
       "2021-06-15    12186.847111\n",
       "2021-06-16    11592.613360\n",
       "2021-06-17    11902.873990\n",
       "2021-06-18    11478.596588\n",
       "2021-06-19    10688.597153\n",
       "2021-06-20     9722.139433\n",
       "2021-06-21     9810.855259\n",
       "2021-06-22     8013.925421\n",
       "2021-06-23     7051.878522\n",
       "2021-06-24     6825.418123\n",
       "2021-06-25     6428.158123\n",
       "2021-06-26     6071.535596\n",
       "Freq: D, dtype: float64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sktime.forecasting.model_selection import temporal_train_test_split\n",
    "y_train, y_test = temporal_train_test_split(y, test_size=24)\n",
    "fh = ForecastingHorizon(y_test.index, is_relative=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Date\n",
       "2022-03-08    861323.0\n",
       "2022-03-09    861468.0\n",
       "2022-03-10    861617.0\n",
       "2022-03-11    861733.0\n",
       "2022-03-12    861879.0\n",
       "2022-03-13    862044.0\n",
       "2022-03-14    862098.0\n",
       "2022-03-15    862226.0\n",
       "2022-03-16    862355.0\n",
       "2022-03-17    862509.0\n",
       "2022-03-18    862660.0\n",
       "2022-03-19    862812.0\n",
       "2022-03-20    862926.0\n",
       "2022-03-21    862978.0\n",
       "2022-03-22    863055.0\n",
       "2022-03-23    863157.0\n",
       "2022-03-24    863223.0\n",
       "2022-03-25    863319.0\n",
       "2022-03-26    863427.0\n",
       "2022-03-27    863538.0\n",
       "2022-03-28    863576.0\n",
       "2022-03-29    863623.0\n",
       "2022-03-30    863704.0\n",
       "2022-03-31    863790.0\n",
       "Freq: D, Name: 12-17yrs, dtype: float64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(<Figure size 1152x288 with 1 Axes>, <AxesSubplot:>)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler, PowerTransformer\n",
    "from sktime.datasets import load_macroeconomic\n",
    "from sktime.forecasting.compose import ForecastingPipeline\n",
    "from sktime.transformations.series.adapt import TabularToSeriesAdaptor\n",
    "from sktime.transformations.series.impute import Imputer\n",
    "\n",
    "from sktime.forecasting.arima import AutoARIMA\n",
    "from sktime.forecasting.compose import ForecastingPipeline\n",
    "forecaster = ForecastingPipeline(\n",
    "    steps=[\n",
    "        (\"imputer\", Imputer(method=\"mean\")),\n",
    "        (\"scale\", TabularToSeriesAdaptor(MinMaxScaler(feature_range=(1, 2)))),\n",
    "        (\"boxcox\", TabularToSeriesAdaptor(PowerTransformer(method=\"box-cox\"))),\n",
    "        (\"forecaster\", AutoARIMA(suppress_warnings=True)),\n",
    "    ]\n",
    ")\n",
    "forecaster.fit(y=y_train)\n",
    "y_pred = forecaster.predict(fh=fh)\n",
    "\n",
    "plot_series( y_pred, y_test, labels=[ \"actual\", \"forecast\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "616.7426352721978"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rmse = np.mean((y_pred - y_test)**2)**.5  # RMSE\n",
    "rmse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2022-03-08    861285.766691\n",
       "2022-03-09    861360.435247\n",
       "2022-03-10    861435.107283\n",
       "2022-03-11    861509.779196\n",
       "2022-03-12    861584.451114\n",
       "2022-03-13    861659.123031\n",
       "2022-03-14    861733.794948\n",
       "2022-03-15    861808.466865\n",
       "2022-03-16    861883.138783\n",
       "2022-03-17    861957.810700\n",
       "2022-03-18    862032.482617\n",
       "2022-03-19    862107.154534\n",
       "2022-03-20    862181.826452\n",
       "2022-03-21    862256.498369\n",
       "2022-03-22    862331.170286\n",
       "2022-03-23    862405.842203\n",
       "2022-03-24    862480.514121\n",
       "2022-03-25    862555.186038\n",
       "2022-03-26    862629.857955\n",
       "2022-03-27    862704.529872\n",
       "2022-03-28    862779.201790\n",
       "2022-03-29    862853.873707\n",
       "2022-03-30    862928.545624\n",
       "2022-03-31    863003.217541\n",
       "Freq: D, dtype: float64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2022-03-08    861323.0\n",
       "2022-03-09    861468.0\n",
       "2022-03-10    861617.0\n",
       "2022-03-11    861733.0\n",
       "2022-03-12    861879.0\n",
       "2022-03-13    862044.0\n",
       "2022-03-14    862098.0\n",
       "2022-03-15    862226.0\n",
       "2022-03-16    862355.0\n",
       "2022-03-17    862509.0\n",
       "2022-03-18    862660.0\n",
       "2022-03-19    862812.0\n",
       "2022-03-20    862926.0\n",
       "2022-03-21    862978.0\n",
       "2022-03-22    863055.0\n",
       "2022-03-23    863157.0\n",
       "2022-03-24    863223.0\n",
       "2022-03-25    863319.0\n",
       "2022-03-26    863427.0\n",
       "2022-03-27    863538.0\n",
       "2022-03-28    863576.0\n",
       "2022-03-29    863623.0\n",
       "2022-03-30    863704.0\n",
       "2022-03-31    863790.0\n",
       "Freq: D, Name: 12-17yrs, dtype: float64"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "Forecasting.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}

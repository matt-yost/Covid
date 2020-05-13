import pprint
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
from lmfit import models
from pandas.plotting import register_matplotlib_converters
from c19all import df_all, sum_by_date, for_country
import c19us
import constants

register_matplotlib_converters()
pp = pprint.pprint

def sum_by_date_for_country(country):
    df = for_country(df_all['deaths'], country)
    df = sum_by_date(df)
    # df['cases'] = df['cases'].diff()
    return df

# def get_diff(df):
#     df = df[['cases']]
#     df_diff = df.diff()
#     df_diff = df_diff.diff()
#     df_diff.loc[df_diff['cases'] >= 0, 'style'] = 'og'
#     df_diff.loc[df_diff['cases'] < 0, 'style'] = 'vr' 
#     pp(df_diff)
#     return df_diff

# Plots a country's cases by day
def plot_cases_data(country):
    df_a = sum_by_date_for_country(country)
    df_a = df_a[['index', 'cases']]
    plt.plot(df_a.index, df_a.cases, color=numpy.random.rand(3,),label=country+' Deaths')

# Prints Past 10 days of new cases for country
def print_new_cases(country):
    df_diff = sum_by_date_for_country(country)
    df_diff.to_csv('data_'+country+'.csv')
    df_diff = df_diff[['cases']]
    pp(df_diff.iloc[-1])
    df_diff = df_diff['cases'].diff()
    pp(country + ' New Deaths:')
    pp(df_diff[-10:])

# Build Power Law model and plot against cases for given country
def build_model(country):
    df_model = sum_by_date_for_country(country)
    df_model = df_model[['index', 'cases']]
    df_style = df_model[['cases']]
    model = models.PowerLawModel()
    params = model.make_params()
    result = model.fit(df_model.cases, params, x=df_model.index.to_list())
    result.plot_fit(xlabel='Days since 2020-01-22', ylabel=country+' Deaths', datafmt='og', fitfmt='r')
    plt.grid(color='w', linestyle='-', linewidth=0.5)
    plt.show()

# Plot setup & function calls
plt.rcParams['figure.figsize'] = [7,4]
mpl.style.use('dark_background')
plt.grid(color='w', linestyle='-', linewidth=0.5)

plot_cases_data('Germany')
plot_cases_data('Spain')
plot_cases_data('Italy')
plot_cases_data('US')
plot_cases_data('France')

plt.legend()
plt.show()

print_new_cases('US')
print_new_cases('Italy')
print_new_cases('Germany')
print_new_cases('France')
print_new_cases('Spain')

build_model('Germany')
build_model('Italy')
build_model('US')

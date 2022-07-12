#!/usr/bin/python3

import cdsapi
from math import radians, cos, sin, asin, sqrt
from statistics import mean
import csv
import numpy as np
import pandas as pd
import netCDF4 as nc
import metpy.calc as mc
from typing import Dict, NamedTuple

'''
A little library developed for wind feild research
Make sure you have a climate data store account, along with the secure key. The
tutorial on how to set up a (free) account is here: https://cds.climate.copernicus.eu/api-how-to 
for information about the historical data set and the variables:
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview 
for information about the projections:
https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=overview 

'''

# this bounding box class helps prevent mis-ordering the lats/lons
class BoundingBox(NamedTuple):
    lat_up: int
    lon_left: int
    lat_down: int
    lon_right: int

# returns a dict of bounding boxes 
# note- I chose to have the name of each bounding box be an int
# so that its easy to iterate through all the locations
def load_from_csv(file_path: str) -> Dict[int, BoundingBox]:
    with open(file_path, newline='\n') as csv_file:
        # TODO how about if there is a header? 
        reader = csv.reader(csv_file)
        return {
            int(location): BoundingBox(int(lat_up), int(lon_left), int(lat_down), int(lon_right))
            for location, lat_up, lon_left, lat_down, lon_right
            in reader
        }


# calculates distance in km between data gridpoints and the actual lat/lon
# use this to see how much loosing decimal places in your original location affects error 
def distance(lat1, lon1, lat2, lon2):
    #convert from degrees to radians
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1)*cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers (3956 miles)
    r = 6371

    return c*r


#this function is specifically for the era5 reanalysis data  
#note that you can play around with the api request inside, like changing the variables retrieved
def get_historical(from_year, to_year, location, area):
    c = cdsapi.Client()

    year = from_year
    file_location = f'./{year}_vars{location}.nc'

    while year < to_year +1:
        c.retrieve(
        'reanalysis-era5-land',
        {
            'format': 'netcdf',
            'area': area,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'year': year,
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'surface_pressure',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
        },
        file_location
        )
        year += 1
        file_location = f'./{year}_vars{location}.nc'

#this downloads historical GFDL model data
#just gets near surface windspeed, but user can do other variables
def get_historical_GFDL(from_year, to_year, location, area):
    c = cdsapi.Client()

    file_location = f'./1980-2020GFDLwind{location}.nc'

    c.retrieve(
    'projections-cmip6',
    {   
    'format': 'zip',
    'temporal_resolution': 'daily', #'monthly',
    'experiment': 'historical',
    'level': 'single_levels',
    'variable': 'near_surface_wind_speed',
    'model': 'gfdl_esm4',
    'date': f'{from_year}-01-01/{to_year}-12-31',
    'area': area,
    },
    file_location)
        
# extracts data, calculates things like relative humidity, 
# and scales certain params from surface to 80 meters
def organize_historical(files, location, decimals=(0,0)):
    
    # the lat/ lon here represents the value after the decimal;
    # For example, if the original lat is 47.53, lat = 5
    (lat, lon) = decimals

    sp = [] #surface pressures
    temp = [] #temperatures 
    dewp = [] #dewpoints
    wv = [] #wind v components
    wu = [] #wind u components
    dates = []

    for file in files:

        data = nc.Dataset(file)
        
        #time units: hours since 1900-01-01 00:00:00.0
        dates.extend(list(nc.num2date(data.variables['time'], 'hours since 1900-01-01 00:00:00.0', 'gregorian', True)))

        surface_pressure = data.variables['sp']#[time: 0-8759][lat: 0-10][lon:0-10]
        temp_2m = data.variables['t2m']
        dewpoint = data.variables['d2m']
        wind_v = data.variables['v10']
        wind_u = data.variables['u10']


        #every hour 
        for hour in range(0,8760):
            sp.append(surface_pressure[hour][lat][lon])
            temp.append(temp_2m[hour][lat][lon])
            dewp.append(dewpoint[hour][lat][lon])
            wv.append(wind_v[hour][lat][lon])
            wu.append(wind_u[hour][lat][lon])

        # writer.writerows(list(zip(dates, sp, temp, dewp, wv, wu)))
        
    dates = dates[:len(sp)]

    # #make a dataframe so we can play with the vars, make new cols
    dictionary = {'date (hourly)': dates, 'surface_pressure (Pa)': sp, '2m_temperature (K)': temp,
    '2m_dewpoint (K)': dewp, 'wind_u_component (m/s)': wu, 'wind_v_component (m/s)': wv}

    df = pd.DataFrame(dictionary)

    # calculate wind 
    df['wind_speed (m/s)'] = (df['wind_v_component (m/s)']**2 + df['wind_u_component (m/s)']**2)**0.5
    # scale wind 
    df['wind_speed_80m (m/s)'] = df['wind_speed (m/s)']*(8)**(1/7) #wind shear exponent is estimated to be 1/7
    #calculate the relative humidity
    df['relative_humidity (%)'] =(np.exp((17.625*df['2m_dewpoint (K)'])/(243.04+df['2m_dewpoint (K)']))/np.exp((17.625*df['2m_temperature (K)'])/(243.04+df['2m_temperature (K)'])))
    
    df.to_csv(f'windfarm{location}data1980-2020.csv')


#here you can get projections for various variables, models, and experiments
def get_projection(from_year, to_year, location, area):
    c = cdsapi.Client()

    # if you want to change temporal resolution, the ssp scenario, variable, model, etc,
    # poke around in this c.retrieve section. On the cds website, you can make your own api request too:
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=overview 
    # go to download tab, put in parameters, and then click "show api request" at the bottom
    c.retrieve(
        'projections-cmip6',
        {
            'format': 'zip',
            'temporal_resolution': 'monthly', #'daily', 
            'experiment': 'ssp2_4_5', #'ssp1_2_6', #'ssp5_8_5', #'ssp2_4_5',
            'level': 'single_levels',
            'variable':'near_surface_wind_speed',# 'near_surface_specific_humidity', #'sea_level_pressure', #'near_surface_wind_speed', #'near_surface_air_temperature',
            'model':'gfdl_esm4',#'bcc_csm2_mr', 'cmcc_esm2', 'cmcc_cm2_sr5', 'ec_earth3_veg', 'inm_cm5_0', 'mri-esm2_0', 'noresm2_mm'
            'date': f'{from_year}-01-01/{to_year}-12-31',
            'area': area
        },
        f'gfdl_wind_prediction{location}.zip')

#same as organize_historical, just different format of data/ needs its own organization 
def organize_projections(files, loc, just_wind=True):
    
    df = pd.DataFrame()
    # just getting the dates
    data = nc.Dataset(files[0])
    df['date'] = nc.num2date(data.variables['time'], 'days since 0001-01-01', 'noleap', True)

    #wind
    predictions = []
    for day in range(len(df['date'])):
        predictions.append(data.variables['sfcWind'][day][0][0])

    df['wind_2m m/s'] = predictions
    #wind_scaled
    df['wind_80m m/s'] = df['wind_2m m/s']*(40)**(1/7)

    if not just_wind:
        #temperature
        data = nc.Dataset(files[1])
        predictions = []
        for day in range(len(df['date'])):
            predictions.append(data.variables['tas'][day][0][0])
        df['surface_temp'] = predictions

        #pressure
        data = nc.Dataset(files[2])
        predictions = []
        for day in range(len(df['date'])):
            predictions.append(data.variables['psl'][day][0][0])
        df['sea_level_pressure'] = predictions

        #humidity
        data = nc.Dataset(files[3])
        predictions = []
        for day in range(len(df['date'])):
            predictions.append(data.variables['huss'][day][0][0])
        df['near_surface_humidity'] = predictions
        #relative humidity
        df['relative_humidity (%)'] = df.apply(lambda x: mc.relative_humidity_from_specific_humidity(mc.units.Quantity(x.sea_level_pressure, "Pa"), mc.units.Quantity(x.surface_temp, "K"), x.near_surface_humidity), axis=1)
        
    df.to_csv(f'projection_gfdl{loc}.csv')

# the following functions are helpful for exploring and visualizing the data

#turns hourly historical data into daily by taking average
def dailify(df):
    d = []
    w = []
    t = []
    p = []
    h = []

    for i, group in df.groupby(np.arange(len(df)) // 24): #taking the average of 24 hour period
        d.append(list(group['date'])[0])
        w.append(mean(group['wind_speed_80m (m/s)']))
        t.append(mean(group['2m_temperature (K)']))
        p.append(mean(group['surface_pressure (Pa)']))
        h.append(mean(group['relative_humidity (%)']))
        

    return pd.DataFrame({'date': d, 'wind_speed_80m': w, 'surface_pressure': p, '2m_temperature': t, 'relative_humidity': h})

#turns relative humidity values to floats so we can do calculations with them
def rh_str_to_num(projection):
    rh = []
    for x in projection['relative_humidity']:
        rh.append(float(x[:7]))

    projection['relative_humidity'] = rh  

#combines data into one csv file and computes power
def combine(historical, projection):
    # power = [(air density) times (swept area of blades) times (wind speed cubed)] divided by 2. The area is in meters squared, air density is in kilograms per meters cubed and wind speed is in meters per second.
    # air pressure at altitude: P = (surface pressure) exp(-9.80665*0.0289644*80/(8.31432(temperature at 80m)))
    # air density is air mass / volume
    # air temperature scaled: T = -131 + (0.003 * altitude in meters)
    # really rough temp conversion: -2 degrees per 304.8 meters; so - .53 degrees for 80m 

    full = pd.DataFrame()

    full['date (daily)'] = historical['date'].append(projection['date'])
    # full['surface_wind (m/s)'] = historical['wind_speed'].append(projection['wind_2m'])
    full['80m_wind (m/s)'] = historical['wind_speed_80m'].append(projection['wind_80m'])
    full['surface_temperature (K)'] = historical['2m_temperature'].append(projection['surface_temp'])
    full['80m_temperature (K)'] = full['surface_temperature (K)'].apply(lambda x: x - .53)
    full['surface_pressure (Pa)'] = historical['surface_pressure'].append(projection['sea_level_pressure'])
    full['80m_pressure (Pa)'] = full['surface_pressure (Pa)']*np.exp(-9.80665*0.0289644*80/(8.31432*(full['80m_temperature (K)'])))
    full['relative_humidity_surface (%)'] = (historical['relative_humidity']/100).append(projection['relative_humidity'])
    full['80m_relative_humidity (%)'] = [min(1, 1.32*x) for x in full['relative_humidity_surface (%)']]

    saturation_vapor_p = 6.1078*10*(7.5*(full['80m_temperature (K)'] + 237.3)/full['80m_temperature (K)'])
    vapor_p = full['80m_relative_humidity (%)']*saturation_vapor_p
    partial_p_dry = full['80m_pressure (Pa)'] - vapor_p
    full['80m_air_density (kg/m^3)'] = partial_p_dry/287.058 + vapor_p/(461.495*full['80m_temperature (K)'])

    #radius of swept area is 50m
    full['80m_power (watt)'] = (full['80m_air_density (kg/m^3)']*(np.pi*(50**2))*full['80m_wind (m/s)']**3)/2

    full.to_csv('wind_farm_4_power.csv')

#pretty self explanatory, use this for sanity checks and very general comparisons
def average_wind(df):
    return sum(df['wind_80m m/s'])/len(df['wind_80m m/s'])



'''
Here are some examples to help you get started:

boxes = load_from_csv('/home/veronika/4brammer/sample_locations.csv')

for i in range(1,4): 
    get_projection(2020,2060, i, boxes.get_area(i))

location_1_data = ['path to wind data for location 1']

organize_projections(location_1_data, 1)

location_2_data = ['path to wind data for location 2', 'path to temp data for location 2', 
'path to pressure data for location 2', 'path to humidity data for location 2']

organize_projections(location_2_data, 2, just_wind=False)

Useful tips:
* if theres an error when parsing the .nc file, try unzipping the file 
(weird, I know. I'm pretty sure this is a bug in era5 reanalysis files)
* a very useful terminal command is h5dump, it niceley displays the structure/ data of a netcdf/hdf5 file
* use Global Wind Atlas for sanity checks 
* this library will take your visualizations to the next level: "from mpl_toolkits.basemap import Basemap"

'''
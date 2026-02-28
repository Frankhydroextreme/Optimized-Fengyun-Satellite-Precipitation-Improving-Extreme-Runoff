import numpy as np
import pandas as pd
import h5py
import os
import re
import time
from awx import Awx
from shapely.geometry import Point
import pickle
from datetime import datetime, timedelta
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor
from math import radians, sin, cos, sqrt, atan2

# ----------------------------
# Core Function Definitions
# ----------------------------
# Get longitude and latitude from the full-disk nominal file longitude/latitude lookup table (Northern Hemisphere)
def getLatLonFromdat(lonlatfile, ref=104.5):
    with open(lonlatfile, 'rb') as f:
        lon_fy = np.fromfile(f, count=2288 * 2288, dtype='float32') + ref  # Store longitude first, add the corresponding longitude value according to the satellite
        lat_fy = np.fromfile(f, count=2288 * 2288, dtype='float32')  # Then store latitude
    # Handle invalid values
    lon_fy[lon_fy == 300 + ref] = np.nan
    lat_fy[lat_fy == 300] = np.nan
    # Extract Northern Hemisphere
    lon = lon_fy[:len(lon_fy)//2]
    lat = lat_fy[:len(lat_fy)//2]
    return lon, lat

# Get longitude and latitude from AWX sample file
def getLatLonOfawx(filepath):
    ds = Awx(pathfile=filepath)
    dar = ds.values.squeeze()
    lon_label = dar.lon.data
    lat_label = dar.lat.data
    lon = np.tile(lon_label, reps=501)
    lat = np.repeat(lat_label, repeats=1001)
    return lon, lat

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the spherical distance between two points (Unit: km)"""
    R = 6371.0  # Earth radius (km)
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

def find_nearest_grid(stations_file, awx_path):
    # Read station data
    stations = pd.read_excel(stations_file)
    station_ids = stations['station_id'].astype(str).values  # Convert to string
    station_lons = stations['longitude'].values
    station_lats = stations['latitude'].values
    # Get longitude and latitude of satellite precipitation grid points
    grid_lons, grid_lats = getLatLonOfawx(awx_path)
    # Store results
    result = {}
    # Find the nearest grid point for each station
    for i in range(len(station_ids)):
        min_distance = float('inf')
        nearest_idx = -1
        
        # Calculate distance from the station to each grid point
        for j in range(len(grid_lons)):
            distance = haversine(station_lons[i], station_lats[i], grid_lons[j], grid_lats[j])
            if distance < min_distance:
                min_distance = distance
                nearest_idx = j
        # Store the result in the dictionary
        result[i] = [np.int64(nearest_idx)]
    return result

def get_extreme_date(filepath):
    # Read Excel file
    df = pd.read_excel(filepath)
    # Calculate the 75% threshold (75th percentile)
    threshold_75 = df['Flow'].quantile(0.70)
    # Filter records where runoff exceeds the 75% threshold
    exceed_threshold = df[df['Flow'] > threshold_75].copy()  # Use .copy() to create a copy
    # Construct date column
    exceed_threshold.loc[:, 'Date'] = pd.to_datetime(exceed_threshold[['Year', 'Month', 'Day']].rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'}))
    # Keep only the date (excluding runoff values)
    dates = exceed_threshold['Date'].dt.strftime('%Y-%m-%d')
    return dates

# Get the index values of grid points in each sub-basin. If there are no grid points in the sub-basin, take the nearest grid point (supports specified ref)
def getIndices(shp_path, lonlatfile, ref, filetype='hdf', awx_path=None):
    if filetype == 'hdf':
        longitude, latitude = getLatLonFromdat(lonlatfile, ref=ref) # Get longitude and latitude from the full-disk nominal file longitude/latitude lookup table (Northern Hemisphere)
    elif filetype == 'awx':
        longitude, latitude = getLatLonOfawx(awx_path)
    else:
        raise ValueError("The 'filetype' parameter must be 'hdf' or 'awx'")
    
    gdf_shp = gpd.read_file(shp_path)
    # Convert lon/lat to GeoDataFrame
    points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(longitude, latitude)],
        index=np.arange(len(longitude)), # Set index
        crs="EPSG:4326" # Lon/lat are in WGS84 coordinate system
    )
    # Ensure the coordinate system of the shapefile and points are consistent
    if gdf_shp.crs != points.crs:
        gdf_shp = gdf_shp.to_crs(points.crs)
    # Build spatial index for points
    sindex = points.sindex
    # Store point indices within each polygon
    indices_results = {}
    # Iterate through each polygon
    for idx, row in gdf_shp.iterrows():
        polygon = row.geometry # Current polygon
        polygon_id = idx  # Can use a specific field in the shapefile as an identifier, e.g., row['id']
        # Filter candidate points using spatial index
        possible_matches_index = list(sindex.intersection(polygon.bounds))
        # Precisely check which points are inside the polygon
        contained_indices = [
            i for i in possible_matches_index if polygon.contains(points.geometry[i])
        ]
        # If no points are inside the polygon, check the bounding box
        if not contained_indices:
            if possible_matches_index:  # Bounding box has candidate points but they are not inside the polygon
                # Calculate the distance from candidate points to the polygon
                distances = [
                    (i, polygon.distance(points.geometry[i]))
                    for i in possible_matches_index
                ]
                nearest_idx = min(distances, key=lambda x: x[1])[0]
                contained_indices = [nearest_idx]
            else:  # No points in the bounding box, expand the bounding box by 0.1 degrees
                minx, miny, maxx, maxy = polygon.bounds
                expanded_bounds = (minx - 0.1, miny - 0.1, maxx + 0.1, maxy + 0.1)
                possible_matches_index = list(sindex.intersection(expanded_bounds))
                if possible_matches_index:   # Candidate points found after expansion
                    distances = [
                        (i, polygon.distance(points.geometry[i]))
                        for i in possible_matches_index
                    ]
                    nearest_idx = min(distances, key=lambda x: x[1])[0]
                    contained_indices = [nearest_idx]
                else:
                    print(f"Polygon {polygon_id} still has no candidate points after expanding by 0.1 degrees")
                    contained_indices = []   # If still no points, return an empty list
        # Store results
        indices_results[polygon_id] = contained_indices
    return indices_results

# Generate and save index tables for all satellite configurations
def generate_all_indices(shp_path, lonlatfile, output_dir='indices_tables'):
    """
    Pre-generate and save sub-basin index tables for all satellite configurations.
    
    Parameters:
        shp_path (str): Shapefile path
        lonlatfile (str): Longitude and latitude file path
        output_dir (str): Directory to save the index tables
    """
    # Define 9 satellite configurations
    satellite_configs = {
        'FY2C_ref_123.5': {'ref': 123.5},
        'FY2D_ref_86.5': {'ref': 86.5},
        'FY2E_ref_105.0': {'ref': 105.0, 'end': '2015-06-02'},
        'FY2E_ref_86.5': {'ref': 86.5, 'start': '2015-07-01'},
        'FY2F_ref_112.5': {'ref': 112.5},
        'FY2G_ref_99.5': {'ref': 99.5, 'end': '2015-05-31'},
        'FY2G_ref_105.0': {'ref': 105.0, 'start': '2015-06-01', 'end': '2018-04-09'},
        'FY2G_ref_99.2': {'ref': 99.2, 'start': '2018-04-16'},
        'FY2H_ref_79.0': {'ref': 79.0}
    }
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate and save index tables
    for config_name, config in satellite_configs.items():
        ref = config['ref']
        indices_results = getIndices(shp_path, lonlatfile, ref, 'hdf')
        output_file = os.path.join(output_dir, f"{config_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(indices_results, f)
        print(f"Index table generated and saved: {config_name}.pkl")

# Load index table
def load_indices(satellite, date, indices_dir='indices_tables'):
    satellite_configs = {
        'FY2C': [{'ref': 123.5}],
        'FY2D': [{'ref': 86.5}],
        'FY2E': [
            {'ref': 105.0, 'end': '2015-06-02'},
            {'ref': 86.5, 'start': '2015-07-01'}
        ],
        'FY2F': [{'ref': 112.5}],
        'FY2G': [
            {'ref': 99.5, 'end': '2015-05-31'},
            {'ref': 105.0, 'start': '2015-06-01', 'end': '2018-04-09'},
            {'ref': 99.2, 'start': '2018-04-16'}
        ],
        'FY2H': [{'ref': 79.0}]
    }
    
    date = pd.to_datetime(date)
    config = satellite_configs.get(satellite)
    if not config:
        raise ValueError(f"Unknown satellite type: {satellite}")
    
    ref = None
    for period in config:
        start = pd.to_datetime(period.get('start', '1900-01-01'))
        end = pd.to_datetime(period.get('end', '9999-12-31'))
        if start <= date <= end:
            ref = period['ref']
            break
    if ref is None:
        raise ValueError(f"Date {date} does not match any time period for {satellite}")
    
    config_name = f"{satellite}_ref_{ref}"
    indices_file = os.path.join(indices_dir, f"{config_name}.pkl")
    if not os.path.exists(indices_file):
        raise FileNotFoundError(f"Index table {indices_file} does not exist, please run generate_all_indices first")
    
    with open(indices_file, 'rb') as f:
        indices_results = pickle.load(f)
    
    return indices_results    

# Get sub-basin representative precipitation from a single file
def getSubPrep(file_path, date, satellite, method='mean', indices_dir='indices_tables', filetype='hdf'):
    """
    Parameters:
        file_path (str): HDF file path
        method (str or callable): Calculation method for sub-basin representative precipitation, optional:
            - 'mean': Mean value (default) - 'max': Maximum value - 'min': Minimum value - 'sum': Sum
            - Or a custom function, accepting a 1D array and returning a scalar
    """
    # Define supported calculation methods
    method_dict = {'mean': np.mean, 'max': np.max, 'min': np.min, 'sum': np.sum}
    # Check the method parameter
    if isinstance(method, str):
        if method not in method_dict:
            raise ValueError(f"Unsupported method value: {method}, optional values are {list(method_dict.keys())} or custom function")
        calc_func = method_dict[method]
    elif callable(method):
        calc_func = method  # If a function is passed in, use it directly
    else:
        raise TypeError("method must be a string or a callable function")
    
    if filetype == 'hdf':
        indices_results = load_indices(satellite, date, indices_dir)
        # Open HDF5 file
        with h5py.File(file_path, 'r') as f:
            # Get all dataset names
            dataset_names = list(f.keys())
            # Find dataset containing 'PRE'
            pre_dataset = None
            for name in dataset_names:
                if 'PRE' in name:
                    pre_dataset = name
                    break  # Exit when the first match is found
            if pre_dataset is None:
                raise KeyError(f"Dataset containing 'PRE' not found in file {file_path}")
            # Read data
            data = f[pre_dataset][:]
    elif filetype == 'awx':
        ds = Awx(pathfile=file_path)
        dar = ds.values.squeeze()
        data = dar.data
        min_lon = dar.ul_lon / 100
        if min_lon == 37:
            indices_results = indices_awx
        elif min_lon == 55:
            indices_results = indices_awx_02
        elif min_lon == 62:
            indices_results = indices_awx_03
        elif min_lon == 30:
            indices_results = indices_awx_04

    # Flatten to 1D array
    data_array = data.flatten()
    # Calculate representative precipitation for each sub-basin (mean value)
    precipitation_results = {}
    for idx, indices in indices_results.items():
        if indices:  # Ensure the index list is not empty
            # Extract precipitation values for grid points within the sub-basin
            precip_values = data_array[indices]
            # Calculate representative precipitation using the specified method
            representative_precip = calc_func(precip_values)
            precipitation_results[idx] = representative_precip
        else:
            # If index is empty, set to NaN or other default values
            precipitation_results[idx] = 0
            print(f"Sub-basin {idx+1} has no grid points, representative precipitation set to 0")
    # Replace all NaNs with 0
    precipitation_results = {k: 0 if np.isnan(v) else v for k, v in precipitation_results.items()}  
    return precipitation_results

# Pre-scan the directory for files and extract dates (24h scale)
# def getallFiles(data_dir): 
#     file_date_map = {}
#     for filename in os.listdir(data_dir):
#         # Use regex to extract date (YYYYMMDD)
#         match = re.search(r'(\d{8})', filename)  # Match 8-digit numbers, e.g., 20170122
#         if match:
#             date_str = match.group(1)
#             file_date_map[date_str] = filename  
#     return file_date_map

def getallFiles(data_dir):
    file_date_map = {}
    for filename in os.listdir(data_dir):
        match = re.search(r'(\d{8})_(\d{4})', filename)  # Match YYYYMMDD_HHMM
        if match:
            date_str = match.group(1)  # Extract date "20170101"
            time_str = match.group(2)[:2]  # Extract time (first two digits only, ignore minutes)
            datetime_str = f"{date_str}_{time_str}"  # Combine to "20170101_01"
            file_date_map[datetime_str] = filename  # Store mapping of date to filename
    return file_date_map

# Iterate through the time range to get sub-basin representative precipitation series and missing times
def getSubPrepSeries(start_date, end_date, filetype, method='mean'):
    # Pre-scan the directory for files and extract dates
    file_date_map = getallFiles(data_dir)
    # Pre-scan the directory for files and extract off-hour dates
    off_hour_map = getallFiles(offdata_dir)
    # Define time range
    time_range = pd.date_range(start=start_date, end=end_date, freq='h')
    # Initialize dictionary for storing precipitation series
    temp_precip_series = {idx: [] for idx in indices_awx}
    missing_times = []  # Record missing file dates
    # Iterate through the time range
    for i, current_time in enumerate(time_range):
        date_str = current_time.strftime('%Y%m%d_%H')  # Construct key with date + hour, ignore minutes, e.g., '20071231_00'
        expected_filename = file_date_map.get(date_str)  # Get the filename for the corresponding date
        is_off_hour = False  # Mark whether it is an off-hour file
        # If the top-of-hour file does not exist, try finding an off-hour file
        if expected_filename is None:
            expected_filename = off_hour_map.get(date_str)
            is_off_hour = True  # Mark as an off-hour file
        if expected_filename:
            # Select directory based on file source
            file_path = os.path.join(offdata_dir if is_off_hour else data_dir, expected_filename)
            # Split filename
            parts = expected_filename.split('_')
            # Extract satellite name (assuming it's always the first part)
            satellite = parts[0]  # 'FY2G'
            # Extract date (assuming it's the second to last part)
            date_raw = parts[-2]  # '20171201'
            # Convert to 'YYYY-MM-DD' using datetime
            date = datetime.strptime(date_raw, '%Y%m%d')
            check_time = current_time + timedelta(hours=7)
            check_date = check_time.strftime('%Y-%m-%d')
            if check_date in extreme_dates.values:
                try:
                    # Extract precipitation for this file
                    precipitation = getSubPrep(file_path, date, satellite, 'max', reffile_dir, filetype)
                    # Append precipitation values to the series
                    for idx, precip_value in precipitation.items():
                        temp_precip_series[idx].append(precip_value)
                except Exception as e:
                    # File processing failed, record date and pad with NaN
                    missing_times.append((current_time, "processing_failed"))
                    print(f"File {expected_filename} processing failed: {e}")
                    for idx in indices_awx:
                        temp_precip_series[idx].append(np.nan)
            else:
                try:
                    # Extract precipitation for this file
                    precipitation = getSubPrep(file_path, date, satellite, 'mean', reffile_dir, filetype)
                    # Append precipitation values to the series
                    for idx, precip_value in precipitation.items():
                        temp_precip_series[idx].append(precip_value)
                except Exception as e:
                    # File processing failed, record date and pad with NaN
                    missing_times.append((current_time, "processing_failed"))
                    print(f"File {expected_filename} processing failed: {e}")
                    for idx in indices_awx:
                        temp_precip_series[idx].append(np.nan)
        else:
            # File not found, record missing date and pad with NaN
            missing_times.append((current_time, "file_not_found"))
            print(f"File for date {date_str} does not exist")
            for idx in indices_awx:
                temp_precip_series[idx].append(np.nan)
        if i % 200 == 0:
            percent = (i / len(time_range)) * 100
            print(f"Progress: {i}/{len(time_range)} ({percent:.1f}%)")
            print(f"Current time: {__import__('datetime').datetime.now()}")
    # Convert lists to pandas.Series with time index
    precip_series = {
        idx: pd.Series(values, index=time_range)
        for idx, values in temp_precip_series.items()
    }
    return precip_series, missing_times

# Interpolation
def interpolate_precip_series(precip_series, method='linear', **kwargs):
    """
    Interpolate NaN values in the sub-basin precipitation series.
    Parameters:
        precip_series (dict): Dictionary with sub-basin IDs as keys and pandas.Series as values
        method (str): Interpolation method, default 'linear', optional values include:
            - 'linear': Linear interpolation
            - 'nearest': Nearest neighbor interpolation
            - 'polynomial': Polynomial interpolation (requires order)
            - 'spline': Spline interpolation (requires order)
            - See pandas.Series.interpolate documentation for more methods
        **kwargs: Additional arguments passed to interpolate (e.g., order)
    Returns:
        dict: Interpolated precip_series
    """
    interpolated_series = {}
    for subbasin_id, series in precip_series.items():
        # Interpolation
        interpolated = series.interpolate(method, **kwargs)
        # Pad NaNs at start/end (preventing unhandled NaNs by interpolate)
        interpolated = interpolated.bfill().ffill()
        interpolated_series[subbasin_id] = interpolated
    return interpolated_series

# Generate daily precipitation (Beijing Time 20:00 - 20:00)
def get_daily_precip_series(hourly_precip_series):
    """
    Calculate the daily precipitation series for each sub-basin. Daily precipitation is defined as the sum from 12:00 the previous day to 11:00 the current day.
    Parameters: hourly_precip_series (dict): Dictionary with sub-basin IDs as keys and hourly time-indexed pandas.Series as values
    Returns: dict: Dictionary with sub-basin IDs as keys and daily time-indexed precipitation pandas.Series as values
    """
    daily_precip_series = {}
    for subbasin_id, hourly_series in hourly_precip_series.items():
        # Ensure the index is a DatetimeIndex
        if not isinstance(hourly_series.index, pd.DatetimeIndex):
            hourly_series.index = pd.to_datetime(hourly_series.index)
        # Get the time range
        start_time = hourly_series.index.min()
        end_time = hourly_series.index.max()
        # Normalize start_time to 00:00 of the next day
        start_date = (start_time + pd.Timedelta(days=1)).floor('D')
        # Normalize end_time to 00:00 of the current day
        end_date = end_time.floor('D')
        # Generate full date range (starting from the second day, since the first day lacks complete previous day data)
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Initialize the daily precipitation list
        daily_values = []
        # Iterate over each date, calculate the sum from 13:00 the previous day to 12:00 the current day
        for date in daily_dates:
            # Define the time window
            window_start = date - pd.Timedelta(hours=11)  # 13:00 previous day
            window_end = date + pd.Timedelta(hours=12)   # 12:00 current day
            # Extract data within the window
            window_data = hourly_series.loc[window_start:window_end]
            # Calculate sum
            if len(window_data) > 0:  # Ensure there is data in the window
                daily_p = window_data.sum()
            else:
                daily_p = np.nan  # Set to NaN if there is no data in the window
            daily_values.append(daily_p)
        # Create Series with date index
        daily_precip_series[subbasin_id] = pd.Series(daily_values, index=daily_dates)
    return daily_precip_series

# Save Functions
def save_single_subbasin01(args):
    subbasin_id, series, time_df, output_dir, label = args
    df = time_df.copy()
    df['Precipitation'] = series.values
    filename = f"{subbasin_id+60001}_{label}_precip.csv"
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)

def save_hourly_precip(precip_series, output_dir, label='interpolated'):
    """
    Save precipitation series using parallel writing, CSV format, and pre-computed time columns.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Pre-compute time columns
    sample_series = next(iter(precip_series.values()))
    time_index = sample_series.index
    time_df = pd.DataFrame({
        'Year': time_index.year,
        'Month': time_index.month,
        'Day': time_index.day,
        'Hour': time_index.hour
    })
    
    # Prepare tasks
    tasks = [(subbasin_id, series, time_df, output_dir, label) for subbasin_id, series in precip_series.items()]
    
    # Save in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(save_single_subbasin01, tasks)
    
    print(f"The {label} hourly precipitation series of sub-basins have been saved to {output_dir}")

def save_single_subbasin02(args):
    """
    Save the daily precipitation series for a single sub-basin.
    
    Parameters:
        args (tuple): (subbasin_id, series, time_df, output_dir)
    """
    subbasin_id, series, time_df, output_dir = args
    df = time_df.copy()
    df['Precipitation'] = series.values
    filename = f"{subbasin_id+60001}.csv"  # Changed to CSV format
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)

def save_daily_precip(precip_series, output_dir):
    """
    Save the daily precipitation series for each sub-basin using parallel writing, CSV format, and pre-computed time columns.
    
    Parameters:
        precip_series (dict): Dictionary with sub-basin IDs as keys and pandas.Series as values
        output_dir (str): Output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Pre-compute time columns (assuming all series have the same time index)
    sample_series = next(iter(precip_series.values()))
    time_index = sample_series.index
    time_df = pd.DataFrame({
        'Year': time_index.year,
        'Month': time_index.month,
        'Day': time_index.day
    })
    
    # Prepare parallel tasks
    tasks = [(subbasin_id, series, time_df, output_dir) for subbasin_id, series in precip_series.items()]
    
    # Use thread pool to save in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(save_single_subbasin02, tasks)
    
    print(f"The daily precipitation series of sub-basins have been saved to {output_dir}")   

def save_missing_times(missing_times, output_dir):
    if missing_times:  # Ensure the list is not empty
        times, reasons = zip(*missing_times)  # Unpack into two lists
    else:
        times, reasons = [], []  # Handle empty list

    # Convert times to pandas Series and parse as datetime
    times_series = pd.Series(times)
    times_series = pd.to_datetime(times_series, errors='coerce')  # Ensure type consistency

    # Create DataFrame containing time, reason, and parsed year, month, day, hour
    df = pd.DataFrame({
        'UTC_Time': times_series,
        'Year': times_series.dt.year,
        'Month': times_series.dt.month,
        'Day': times_series.dt.day,
        'Hour': times_series.dt.hour,
        'Reason': reasons  # Reason column
    })
    # Save to CSV file
    df.to_csv(os.path.join(output_dir, 'missing_times.csv'), index=False)

# ----------------------------
# Configuration Parameters (Please modify according to the actual situation)
# ----------------------------       
# Input key paths
data_dir = r'\InputData' # Path where HDF files are located
offdata_dir = r'\InputData\NonInt' # Path where off-hour files are located
output_dir = r'\OutputData' # Output folder
shp_path = r'\subbasin.shp' # Sub-basin shapefile path 125
lonlatfile = r'\PREDataFY\NOM_ITG_2288_2288(0E0N)_LE\NOM_ITG_2288_2288(0E0N)_LE.dat' # Location of the longitude and latitude lookup table
# station_file = r'\stations.xlsx' # Station longitude and latitude
awx_path = r'\InputData\FY2C_PRE_001_OTG_20071231_1200.AWX' # AWX sample file path
awx_path01 = r'\FY2C_PRE_001_OTG_20071231_1200.AWX' # AWX sample file path 37-137
awx_path02 = r'\FY2C_PRE_001_OTG_20090519_0300.AWX' # AWX sample file path 55-155
awx_path03 = r'\FY2F_PRE_001_OTG_20151202_1000.AWX' # AWX sample file path 62-162
awx_path04 = r'\FY2H_PRE_001_OTG_20181231_0000.AWX' # AWX sample file path 62-162
runoff_path = r'\径流数据.xlsx' # Runoff data
# Construct reffile folder path to store sub-basin grid indices for different satellites
reffile_dir = os.path.join(output_dir, 'reffile')
raw_dir = os.path.join(output_dir, 'raw_hourly_precip')
interpolated_dir = os.path.join(output_dir, 'interpolated_hourly_precip')
daily_dir = os.path.join(output_dir, 'daily_precip')
daily_dir_inter = os.path.join(output_dir, 'daily_precip_inter')
# Define start time
start_time = '2007-12-31 13:00'
end_time = '2017-12-31 12:00'
# Select the method for sub-basin representative precipitation and missing value interpolation
weighted_method = 'mean'
interpolation_method = 'linear'
filetype = 'awx'

# ----------------------------
# Main Program Execution
# ----------------------------
if __name__ == "__main__":
    # Create output folders
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reffile_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(interpolated_dir, exist_ok=True)
    os.makedirs(daily_dir, exist_ok=True)
    os.makedirs(daily_dir_inter, exist_ok=True)
    
    elapsed_time = []
    # Load sub-basin shapefile
    print("Step 1: Loading data...")
    start = time.time()
    gdf_shp = gpd.read_file(shp_path)
    elapsed_time.append(time.time() - start)
    
    # Generate and save index tables for all satellite configurations, saved under output_dir\reffile
    print("Step 2: Generating index tables...")
    start = time.time()
    # HDF files
    # generate_all_indices(shp_path, lonlatfile, reffile_dir) 
    # AWX files
    indices_awx = getIndices(shp_path, lonlatfile, 0, filetype, awx_path01)
    indices_awx_02 = getIndices(shp_path, lonlatfile, 0, filetype, awx_path02)
    indices_awx_03 = getIndices(shp_path, lonlatfile, 0, filetype, awx_path03)
    indices_awx_04 = getIndices(shp_path, lonlatfile, 0, filetype, awx_path04)
    # Specify longitude and latitude points
    # indices_awx = find_nearest_grid(station_file, awx_path01)
    # indices_awx_02 = find_nearest_grid(station_file, awx_path02)
    # indices_awx_03 = find_nearest_grid(station_file, awx_path03)
    # indices_awx_04 = find_nearest_grid(station_file, awx_path04)
    elapsed_time.append(time.time() - start)
    
    # Iterate through the time range to get precipitation series and missing times
    print("Step 3: Iterating through the time range to get precipitation series and missing times...")
    start = time.time()
    extreme_dates = get_extreme_date(runoff_path) # Get dates of extreme runoff
    precip_series, missing_times = getSubPrepSeries(start_time, end_time, filetype, weighted_method)
    elapsed_time.append(time.time() - start)
    
    # Interpolation processing
    print("Step 4: Interpolation processing...")
    start = time.time()
    interpolated_precip = interpolate_precip_series(precip_series, interpolation_method)
    elapsed_time.append(time.time() - start)
    
    # Generating daily precipitation series
    print("Step 5: Generating daily precipitation series...")
    start = time.time()
    daily_precip = get_daily_precip_series(precip_series)
    daily_precip_inter = get_daily_precip_series(interpolated_precip)
    elapsed_time.append(time.time() - start)
    
    # Saving precipitation data
    print("Step 6: Saving precipitation data...")
    start = time.time()
    # Save raw hourly precipitation series
    save_hourly_precip(precip_series, raw_dir, label='raw')
    # Save interpolated hourly precipitation series
    save_hourly_precip(interpolated_precip, interpolated_dir, label='interpolated')
    # Save daily precipitation series after timezone conversion
    save_daily_precip(daily_precip, daily_dir)
    save_daily_precip(daily_precip_inter, daily_dir_inter)
    # Save missing time records
    save_missing_times(missing_times, output_dir)
    elapsed_time.append(time.time() - start)
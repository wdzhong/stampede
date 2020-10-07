"""
Look for duration between points

for each road segment, we only have one feature, which
will make it hard to use the interpolation directly
since we also consider the relationship between
features. Thus, we can add the traffic speed feature
in the other direction as the second feature for the
road segment.

The data format to save will be:
for each road segment, for each time interval, record
the average speed.

How about adding other motion sensor data?

- what the time interval should we set? Also every 5
minutes?
- Use the time at the beginning of the road segment
as the time of the corresponding time interval?
-

Written and tested in Python 3.7
"""

import os
import pandas as pd
from datetime import datetime
import pickle as pk
from collections import defaultdict
import json
from typing import List
import bisect
import pathlib
import pdb
from itertools import combinations

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from geopy.distance import distance

from logger import Logger

DEBUG = True
_logger = Logger("logs")


class LatLng:
    """
    Latitude and Longitude pair
    """

    def __init__(self, lat: float, lng: float):
        self.lat = lat
        self.lng = lng

    def to_tuple(self):
        return (self.lat, self.lng)

    def __str__(self):
        return ",".join([str(self.lat), str(self.lng)])


class SrcDst:
    """
    Source and Destination pair
    """

    def __init__(self, source: LatLng, dest: LatLng, source_name: str, dest_name: str):
        """
        Source and destination

        The details of a source and destination pair

        Parameters
        ----------
        source : LatLng obj
            The LatLng of the source point

        dest : LatLng
            The LatLng of the destination point

        source_name : str
            The name of the source

        dest_name : str
            The name of the destination
        """

        self.source = source
        self.dest = dest
        self.source_name = source_name
        self.dest_name = dest_name

    def name(self):
        return self.source_name + "-" + self.dest_name

    def __str__(self):
        return "source: " + str(self.source) + ", destination: " + str(self.dest)


class Path:
    def __init__(self, source: LatLng, dest: LatLng):
        self.source = source
        self.dest = dest
        self.path = []
        self.time = None


def distance_between_gps(point1: LatLng, point2: LatLng) -> float:
    """
    Get the distance between two GPS points. Unit: meters

    Parameters
    ----------
    point1 : LatLng

    point2 : latLng

    Returns
    -------
    distance : float
        in meters
    """
    return distance(point1.to_tuple(), point2.to_tuple()).meters


def read_gps(filename) -> pd.DataFrame:
    """
    Read GPS file and return a DataFrame obj

    Parameters
    -----------
    filename : str
        The full path of the gps file

    Return
    ------
    df : DataFrame
    """
    df = pd.read_csv(filename, sep=',', error_bad_lines=False, skipfooter=1)

    # the gps file may NOT have header, due to the error in generating the first gps file
    # df = df[df['provider'] == 'gps']
    df = df[df.iloc[:, -1] == 'gps']
    # print(df.head(5))
    # print(df.tail(5))
    return df


def look_for_duration(df: pd.DataFrame, sd: SrcDst, max_possible_duration: int = None):
    """
    Look for all gps traces from given gps points that fall onto the specified
    pair of source and destination.

    Parameters
    ----------
    df : DataFrame
        The DataFrame obj that contains GPS data which should not contain 'network'
        provider at this point.

    sd : SrcDst
        The source and destination of the path to look for

    max_possible_duration : int
        The maximum possible duration for the given source destination.

    Return
    -------
    traces : array, element=(start_line, end_line)
        pairs of start_line and end_line from the gps file for each trip
    """

    traces = []

    # TODO: how to locate the source and destination in the GPS data
    # TODO: how to tell when the bus/car arrives and departs
    # straightforward way is to set a threshold, e.g. 10 meters
    gps_np = df.to_numpy()
    source = sd.source
    dest = sd.dest
    threshold = 30  # meters
    data_around_station = []
    """
    for target in [source, dest]:
        # for line in gps_np:
        #     point = LatLng(line[2], line[3])
        #     if distance_between_gps(target, point) <= threshold:
        #         data_around_station.append(line)
        # for index, line in df.iterrows():
        #     point = LatLng(line[2], line[3])
        #     if distance_between_gps(target, point) <= threshold:
        #         data_around_station.append(line)
        for line in df.itertuples(index=False):
            point = LatLng(line.lat, line.lon)
            if distance_between_gps(target, point) <= threshold:
                data_around_station.append(line)

    new_df = pd.DataFrame(data_around_station, columns=df.columns)
    new_df.to_csv('gps_around_station.csv', index=False)
    """

    # TODO: how to tell whether the gps trace is from source to destination directly
    # rather than passing some other places first
    # how to determine the direction is from source to dest, or from dest to source?
    # especially under conditions that there are more than one round trips?
    # calculate the distance between points (e.g. after the source) with the source and
    # destination. Generally, the former one should get larger and larger, and the later
    # should get smaller and smaller
    # Besides, we can also make use of the bearing data
    # do we need to add some time constraint
    trips = []
    start_end_line_numbers = []
    i = 0
    while i < len(gps_np):
        # find the point departs source
        line = gps_np[i]
        point = LatLng(line[2], line[3])
        while i + 1 < len(gps_np) and distance_between_gps(source, point) > threshold:
            i += 1
            line = gps_np[i]
            point = LatLng(line[2], line[3])

        if i >= len(gps_np):
            break

        while i + 1 < len(gps_np) and distance_between_gps(source, point) <= threshold:
            i += 1
            line = gps_np[i]
            point = LatLng(line[2], line[3])

        if i >= len(gps_np):
            break

        start_point = line
        start_line_number = i - 1

        # we need to make sure that the direction is from the source to destination
        # instead of moving away from the destination
        # it is also possible that point "shaking"
        # for the next 50 points, if the majority of points are getting closer to the
        # destination, then we treat the start point as a good one
        initial_dist_to_dest = distance_between_gps(point, dest)
        # the (absolute) number of points that are getting closer to dest
        closer = 0
        count = 0
        while i + 1 < len(gps_np) and count < 50:  # TODO: update the threshold
            i += 1
            line = gps_np[i]
            point = LatLng(line[2], line[3])
            new_dist = distance_between_gps(point, dest)
            if new_dist <= initial_dist_to_dest:
                closer += 1
            else:
                closer -= 1
            count += 1

        if closer <= 0:
            # print("wrong direction")
            i += 1
            continue

        # TODO: it is possible that the bus will pass the source station
        # again after it left, even if it was moving closer to the destination
        # in some way, e.g. from the service center to West

        # find the point arrives at destination
        # TODO: actually, we can just calculate the distance from destination
        # when the distance becomes the minimum, we know that the car has arrived
        # or stayed for a while
        if i >= len(gps_np):
            break

        line = gps_np[i]
        point = LatLng(line[2], line[3])
        repass = False  # indicator if the bus passed the source again
        while i + 1 < len(gps_np) and distance_between_gps(dest, point) > threshold:
            if distance_between_gps(source, point) < threshold:
                # TODO: this is not enough
                # because the threshold might be too small when
                # data missing happens near the stop
                # so we might need to resample/interpolate
                repass = True
                break
            i += 1
            line = gps_np[i]
            point = LatLng(line[2], line[3])
        if repass:
            continue

        if i < len(gps_np):
            # we find one good candidate for destination
            dest_point = line
            trips.append((start_point, dest_point))

            end_line_number = i - 1
            start_end_line_numbers.append((start_line_number, end_line_number))

    for i, (start, end) in enumerate(trips):
        duration = (end[1] - start[1]) // 1000
        _logger.print(f'{datetime.fromtimestamp(start[1] / 1000.0)}, duration (seconds): {duration}')

        if max_possible_duration and duration >= max_possible_duration:
            # save info to log file
            # otherwise, it is hard to figure out this message when there are lots of trips
            _logger.print("    longer than maximum possible duration %d. Need double check. Ignore it." %
                          max_possible_duration)
            _logger.print("    start and end line numbers: %d, %d" %
                          (start_end_line_numbers[i][0], start_end_line_numbers[i][1]))
            continue
        traces.append((start, end))

    return traces


def get_durations(root: str, src_dests: List[SrcDst], output_file: str, road_timestamp_values_file: str,
                  interval_size: int, road_real_time_values_file: str):
    """
    Get trip durations.

    Get trip durations for all (source, destination) pairs for all trips
    under root, and save the result to <output_file>.

    Parameters
    ----------
    root : str
        The path that data is saved

    src_dests : list[SrcDst]
        A list of the SrcDst

    output_file : str
        The file to store the result

    road_timestamp_values_file : str
        The file that contains {road: {timestamp: [running duration]}}

    interval_size: int
        split the timeline into non overlapping intervals, each interval size is 900s, i.e., 15 minutes
    """
    _logger.print(f"interval size is: {interval_size}")

    # {trip_folder: {source_dest: [(start_line, end_line)]}}
    trip_start_end = defaultdict(lambda: defaultdict(list))
    # {road_segment: {time: [list of running speed or travel time]}}
    road_timestamp_values = defaultdict(lambda: defaultdict(list))
    road_real_time_values = defaultdict(list)
    for parent, _, _ in tqdm(os.walk(root)):
        gps_file = os.path.join(parent, "gps_merged.txt")
        if os.path.isfile(gps_file):

            if os.stat(gps_file).st_size == 0:
                print("\tEmpty file.")
                continue

            df = read_gps(gps_file)

            # skip files that have small number of gps readings
            if len(df) < 1000:
                continue

            # print("deal with: %s" % parent)
            _logger.print(f"deal with: {parent}")

            for src_dest in tqdm(src_dests):
                # print(src_dest)

                # TODO: put the max possible time in a dict?
                start_end_pairs = look_for_duration(df, src_dest, 1000)
                trip_start_end[parent][src_dest.name()] = start_end_pairs

                for pair in start_end_pairs:
                    start_time = pair[0][1]
                    end_time = pair[1][1]
                    duration = (end_time - start_time) / 1000  # convert to seconds
                    # TODO: calculate the speed according to the time and distance

                    # get the time interval from the start_time
                    interval = get_timeinterval(start_time, interval_size)
                    road_timestamp_values[src_dest.name()][interval].append(duration)

                    # we should save the actual/real timestamp (i.e., start_time)
                    # so that when we want to try different interval_size, we don't need
                    # to reprocess the raw gps file again which is very slow
                    road_real_time_values[src_dest.name()].append((start_time, duration))

    with open(output_file, "wb") as fp:
        pk.dump(dict(trip_start_end), fp)

    with open(road_timestamp_values_file, "w") as fp:
        json.dump(road_timestamp_values, fp, indent=4)

    with open(road_real_time_values_file, "w") as fp:
        json.dump(road_real_time_values, fp, indent=2)


def get_timeinterval(timestamp: int, interval_size: int = 900):
    """
    Get the corresponding timeinterval for a given timestamp.

    The result depends on how the time intervals are divided, e.g., every 5 minutes,
    every 15 minutes, etc. The goal of this function is to get the interval that
    the given timestamp is in. E.g., if the time interval is every 5 minutes, then
    timestamp 1580923687289 (which is datetime(2020, 2, 5, 12, 28, 7, 289000))
    should be placed in interval datetime(2020, 2, 5, 12, 25), which is
    1580923500. We just need to find the maximum number that is no larger than
    1580923687 (the last three digits should be removed first to get the seconds),
    which is dividable by 300 (5 min * 60 seconds/min).

    Parameters
    ----------
    timestamp: int
        The system timestamp

    interval_size: int, default=900
        The number of seconds for each interval

    Returns
    -------
    timestamp
    """
    t = timestamp // 1000

    # To speed up the looking process
    interval_size_order = 1
    while interval_size % 10 == 0:
        interval_size_order *= 10
        interval_size //= 10

    t //= interval_size_order

    while t % interval_size:
        t -= 1

    t *= interval_size_order
    return t


def load_durations(filename) -> dict:
    """
    Load the durations

    Parameters
    ----------
    filename : str
        File that stores the durations
    """

    with open(filename, "rb") as fp:
        trip_start_end = pk.load(fp)

    '''
    for trip, station_ods in trip_start_end.items():
        print('------------------------')
        print(trip)
        for station, start_end_pairs in station_ods.items():
            print('\t', station)
            for start, end in start_end_pairs:
                print('\t\t', datetime.fromtimestamp(
                    start[1] / 1000.0), end[1] - start[1])
    '''

    return trip_start_end


def deal_with_durations(trip_start_end: dict, root: str):
    """
    Analyse the durations

    Parameters
    ----------
    trip_start_end : dict
        {trip: {source_dest: [(start_line_of_gps_file, end_line)]}},
        where trip is the full path of the trip (TODO: it is not portable),
        source_dest is a string that contains the name of the source station
        and the destination station, e.g. "service_maynard".

    root : str
        The path to save analysed results
    """

    trip_duration = defaultdict(list)

    time_range = get_time_range("00:00", "23:59", "30min")
    segmented_trip_duration = defaultdict(lambda: defaultdict(list))  # {source_dest: {time: [durations]}}
    od_day_time_durations = []

    for trip, station_ods in trip_start_end.items():
        for source_dest, start_end_pairs in station_ods.items():
            for start, end in start_end_pairs:
                trip_start_time = datetime.fromtimestamp(start[1] / 1000.0)
                weekday = trip_start_time.date().weekday()  # Monday is 0, and Sunday is 6
                duration = int((end[1] - start[1]) // 1000)
                trip_duration[source_dest].append([trip_start_time, duration])

                # find out the time slot that the trip falls into
                cur_time = trip_start_time.time()
                insert_slot = bisect.bisect_right(time_range, cur_time) - 1
                # print(cur_time, end=" -- ")
                # print(time_range[insert_slot])
                segmented_trip_duration[source_dest][time_range[insert_slot]].append(duration)
                od_day_time_durations.append([source_dest, weekday, time_range[insert_slot], duration])

    for source_dest, start_duration in trip_duration.items():
        df = pd.DataFrame(start_duration, columns=['start', 'duration'])
        df = df.sort_values(['start'])
        # print(df.head(30))
        # print()
        # print(df.tail(30))
        times = pd.to_datetime(df.start)
        d = df.groupby([times.dt.weekday]).mean()
        # d = df.groupby([times.dt.weekday, times.dt.hour]).mean()
        # d = df.groupby([df['start'], pd.TimeGrouper(freq='30Min')])
        print(d.head(200))
        # d.to_csv(source_dest + '_trip_grouped.csv', index=False)

    for source_dest, time_durations in segmented_trip_duration.items():
        print(source_dest)
        for time_slot, durations in time_durations.items():
            print(time_slot, end=": ")
            print(durations)

    df = pd.DataFrame(od_day_time_durations, columns=['source_dest', 'weekday', 'time', 'duration'])
    df.to_excel(os.path.join(root, 'od_day_time_durations.xlsx'), index=False)

    grouped_daily = df.groupby([df.source_dest, df.weekday]).mean()
    print(grouped_daily.head(14))
    grouped_daily.reset_index()[['source_dest', 'weekday', 'duration']].to_excel(
        os.path.join(root, "od_day_duration_mean.xlsx"), index=False)

    # group by source_dest and time, i.e. don't care weekday
    grouped_hourly = df.groupby([df.source_dest, df.time]).mean()
    # print(grouped_hourly.head(20))
    grouped_hourly = grouped_hourly.reset_index()[['source_dest', 'time', 'duration']]
    grouped_hourly.to_excel(os.path.join(root, "od_time_duration_mean.xlsx"), index=False)

    # group by most details
    grouped_detail = df.groupby([df.source_dest, df.weekday, df.time]).mean()
    # print(grouped_detail.head(30))
    # print(grouped_detail.tail(20))
    grouped_detail = grouped_detail.reset_index()
    grouped_detail.to_excel(os.path.join(root, "od_day_time_duration_mean.xlsx"), index=False)
    plot_od_day_time_duration(grouped_detail.values)


def plot_od_day_time_duration(od_day_time_duration: dict):
    """
    For each source_dest, plot the trip duration of each time slot for each day

    Parameters
    ----------
    od_day_time_duration : dict
        {od: {day: {time: duration}}}
    """

    dict_od_day_time_duration = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for line in od_day_time_duration:
        dict_od_day_time_duration[line[0]][line[1]][line[2]] = line[3]

    for source_dest, day_time_duration in dict_od_day_time_duration.items():
        plot_day_time_duration(day_time_duration, source_dest)


def plot_day_time_duration(day_time_duration: dict, title: str):
    """
    Plot trip duration of each time slot for each day

    Parameters
    ----------
    day_time_duration : dict
        {day: {time: duration}}
    title : str
        The title used in plot
    """

    for day, time_duration in day_time_duration.items():
        times = []
        durations = []
        for time in sorted(time_duration):
            times.append(time)
            durations.append(time_duration[time])
        plt.plot(times, durations, '*', label=day)
        plt.title(title)

    plt.legend()
    plt.show()


def get_time_range(start, end, freq):
    """
    Get time range

    Get time range using given time span and frequency

    Parameters
    ----------
    start : Datetime-style str
        The start time
    end : Datetime-style str
        The end time
    freq : str
        The duration of each time segment

    Returns
    -------
    array
        A list of time object
    """

    time_range = pd.date_range(start=start, end=end, freq=freq).time
    return time_range


def get_stops(filename=None):
    """
    Get information of all stops from the file

    Paramters
    ---------
    filename: str
        The path of the file that contains bus stops information

    Returns
    -------
    stops : dict
        {stop_id: {'name', 'lat', 'lon'}}
    """

    if not filename:
        filename = os.path.join(os.getcwd(), *['data', 'stops.txt'])

    assert os.path.isfile(filename), f"{filename} is not a file."

    stops = {}
    with open(filename, 'r') as f:
        _ = f.readline()
        for line in f.readlines():
            columns = line.rstrip().split(',')  # number,name,lat,lon,direction,place ID on the map
            # print(columns)

            if columns[2] == '' or columns[3] == '':
                continue

            stops[int(columns[0])] = {'name': columns[1], 'lat': float(columns[2]), 'lon': float(columns[3])}

    # print(stops)
    return stops


def load_graph(graph_file):
    """
    Load the graph, i.e., edges (stop id, stop id)

    Parameters
    ----------
    graph_file : str
        The path of the graph file. Each line of the file is a pair of station ids
        separated by a comma

    Returns
    -------
    list:
        Each element is a edge in the graph, i.e. tuple (start, end)
    """
    edges = []

    with open(graph_file, 'r') as f:
        for line in f:
            ids = line.rstrip().split(',')
            edges.append((int(ids[0]), int(ids[1])))

    return edges


def get_road_segments(stops, graph_file):
    """
    Get the road segments from stops and graph

    Parameters
    ----------
    stops : dict
        {stop_id: {'name', 'lat', 'lon'}}

    graph_file : str
        The file that contains the conjunction matrix.

    Returns
    -------
    Array of SrcDst objects
    """
    edges = load_graph(graph_file)

    src_dests = []
    for edge in edges:
        if edge[0] in stops and edge[1] in stops:
            src = stops[edge[0]]
            dest = stops[edge[1]]
            src_dest = SrcDst(LatLng(src['lat'], src['lon']),
                              LatLng(dest['lat'], dest['lon']),
                              src['name'], dest['name'])
            src_dests.append(src_dest)
    return src_dests


def average_interval(input_filename, output_file):
    with open(input_filename, 'r') as fp:
        data = json.load(fp)
        average_data = {}
        for src_dest, time_values in data.items():
            for t, values in time_values.items():
                if src_dest not in average_data:
                    average_data[src_dest] = []
                average_data[src_dest].append([int(t), round(sum(values) / len(values), ndigits=2)])

            average_data[src_dest].sort(key=lambda arr: arr[0])

    with open(output_file, 'w') as fp:
        json.dump(average_data, fp, indent=4)


if __name__ == "__main__":
    interval_size = 300  # in seconds
    lookback_len = 8
    forecasting_len = 3
    missing_rate = 0.6
    mask_rate = 0.2
    spatial_cluster_number = 3
    temporal_cluster_number = 8

    # can use this for quick test, instead of loading all src_dests
    # service_center = LatLng(42.99277, -78.792362)
    # maynard = LatLng(42.96651, -78.810812)
    # src_dests.append(SrcDst(service_center, maynard, 'service', 'maynard'))
    # src_dests.append(SrcDst(maynard, service_center, 'maynard', 'service'))

    root_path = pathlib.Path(".") / "stampede-gps-selected"

    stops_file = root_path / "stops.txt"
    stops = get_stops(stops_file)
    graph_file = root_path / "graph.txt"
    src_dests = get_road_segments(stops, str(graph_file))  # goodyear is not used for now since it is kind of one way

    data_path = root_path / "20200201_20200229"

    result_path = data_path / str(interval_size)

    if not result_path.is_dir():
        result_path.mkdir()

    output_file = result_path / "trip_start_end.pickle"
    # {road: {interval: [values]}}
    road_timestamp_values_file = result_path / "road_timestamp_values.json"

    # {road: [(time, value)]}
    road_realtime_values_file = result_path / "road_realtime_values.json"

    if not road_timestamp_values_file.is_file():
        get_durations(str(data_path), src_dests, str(output_file), str(road_timestamp_values_file), interval_size,
                      road_realtime_values_file)

    trip_start_end = load_durations(output_file)
    # deal_with_durations(trip_start_end, root)

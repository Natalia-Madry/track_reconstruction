import pandas as pd
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_downstream_tracks(path):
    df=pd.read_csv(path)

    #string lists in input data
    list_data=[ "scifi_x","scifi_z","scifi_dxdy",
               "ut_x_min","ut_x_max","ut_y_begin","ut_y_end", "ut_z_position"]

    #change strings to list
    for list_column in list_data:
        df[list_column]=df[list_column].apply(lambda x: ast.literal_eval(x))

    track_data=[]

    for id, row in df.iterrows():
        track_event={
        "track_id":row["track_id"],
        "momentum":row["momentum"], #
        "nSciFiHits":row["nSciFiHits"],
        "nUTHits":row["nUTHits"],
        "isDownstreamTrack": bool(row["isDownstreamTrack"]),

        "scifi_x": row["scifi_x"],
        "scifi_z":row["scifi_z"],
        "scifi_slope_dxdz": row["scifi_slope_dxdz"],
        "scifi_slope_dydz": row["scifi_slope_dydz"],
        "x_position":row["x_position"],
        "y_position":row["y_position"],  #1 number y0 scifi position

        "ut_x_min":row["ut_x_min"],
        "ut_x_max":row["ut_x_max"],
        "ut_y_begin":row["ut_y_begin"],
        "ut_y_end":row["ut_y_end"],
        "ut_z_position": row["ut_z_position"]
        }
        track_data.append(track_event)
    return track_data

def calc_avg_ut_xy_position(events):
    for event in events:
        ut_x=[]
        ut_y=[]

        for ut_x_min, ut_x_max, ut_y_begin, ut_y_end in zip(event["ut_x_min"],event["ut_x_max"],
            event["ut_y_begin"],event["ut_y_end"]):
            ut_x.append((ut_x_min+ut_x_max)/2)
            ut_y.append((ut_y_begin+ut_y_end)/2)

        event["ut_x"]=ut_x
        event["ut_y"]=ut_y
    return events

if __name__=="__main__":
    #load tracks
    events=load_downstream_tracks("sample_small_new_data.csv")
    logging.info("\nEvents number: %d", len(events))
    logging.info(events[0])
    logging.info("\n \n")

    #calculate avarage ut, scifi
    events=calc_avg_ut_xy_position(events)
    logging.info("Events number: %d", len(events))
    logging.info(events[0])









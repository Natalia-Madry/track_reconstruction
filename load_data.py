import torch
import logging
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from build_graph import build_graph_from_track
from read_data import load_downstream_tracks, calc_avg_ut_xy_position

logging.basicConfig(level=logging.INFO, format='%(message)s')

class TrackData(Dataset): #TrackData - changes track to torch geometric graph
    def __init__(self, events, n_false=3):
        super().__init__() #Dataset constructor
        self.tracks=[]
        n_events=len(events)
        for i, ut_track in enumerate(events):
            self.tracks.append({"ut":ut_track, "scifi": ut_track, "label":1, "ut_id":i})

            false_ids=[k for k in range(n_events) if k!=i ] #i==correct id, k=list of wrong ids
            rand_false_ids=torch.randperm(len(false_ids))[:n_false]
            for id in rand_false_ids:
                k=false_ids[id]
                self.tracks.append({"ut":ut_track, "scifi": events[k], "label":0, "ut_id":i})

    def len(self):
        return len(self.tracks)

    def get(self, id):
        track=self.tracks[id]
        graph=build_graph_from_track(ut_track=track["ut"], scifi_tracks=[track["scifi"]], label=track["label"])
        graph.ut_id=torch.tensor([track["ut_id"]])
        return graph

if __name__=="__main__":
    events=load_downstream_tracks("sample_small_new_data.csv")
    events=calc_avg_ut_xy_position(events)

    batch_size=6
    data =TrackData(events)
    data_loader=DataLoader(data, batch_size)

    logging.info("Loaded data converted to graphs: ")
    for batch in data_loader:
        logging.info(batch)

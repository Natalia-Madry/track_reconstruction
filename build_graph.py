import torch, os
from torch_geometric.data import Data
import logging
import matplotlib.pyplot as plt
import random
from read_data import load_downstream_tracks, calc_avg_ut_xy_position

logging.basicConfig(level=logging.INFO, format='%(message)s')

#chart fontsize
plt.rcParams.update({
    "font.size":10,
    "axes.titlesize": 12,
    "axes.labelsize":11,
    "legend.fontsize":10,
    "xtick.labelsize":10,
    "ytick.labelsize": 10,})


def build_graph_from_track(ut_track, scifi_tracks, label):
    #NODES
    node_ft=[]

    det_type={"UT":0, "SciFi":1}
    ut_z0=ut_track["ut_z_position"][0]
    ut_y0=ut_track["y_position"]
    ut_dydz=ut_track["scifi_slope_dydz"]

    #graph: [x, y, z,y_range, ut_y_error, scifi_hit_dist, detector_id]
    for ut_x, ut_y_begin, ut_y_end, ut_z in zip(ut_track["ut_x"], ut_track["ut_y_begin"], ut_track["ut_y_end"], ut_track["ut_z_position"]):
        ut_y_pred=ut_y0+ut_dydz*(ut_z-ut_z0)
        ut_y_range=abs(ut_y_end-ut_y_begin)

        #compute if ut_y_pred is in the ut_y_range
        if ut_y_pred<ut_y_begin:
            ut_y_error=ut_y_begin-ut_y_pred
        elif ut_y_pred>ut_y_end:
            ut_y_error=ut_y_pred-ut_y_end
        else:
            ut_y_error=0.0
        ut_y_error=ut_y_error/(ut_y_range+1e-8) #avoiding division by 0 (if ut_y_range=0 bc of data error- without experienced nan during training)

        node_ft.append([ut_x, ut_y_pred, ut_z, ut_y_error ,ut_y_range,det_type["UT"]]) #features of nodes

    ut_start=0
    ut_end=len(node_ft)-1
    edge_list=[]

    for i in range(ut_start, ut_end):
        edge_list.append([i, i+1]) #edges in UT segment

    for scifi_track in scifi_tracks:
        scifi_start=len(node_ft)
        scifi_z0=scifi_track["ut_z_position"][0]
        scifi_y0=scifi_track["y_position"]
        scifi_dydz=scifi_track["scifi_slope_dydz"]

        for scifi_x, scifi_z in zip(scifi_track["scifi_x"], scifi_track["scifi_z"]):
            scifi_y=scifi_y0+scifi_dydz*(scifi_z-scifi_z0)

            node_ft.append([scifi_x,scifi_y, scifi_z,0,0,det_type["SciFi"]])
        scifi_end=len(node_ft)-1

        for i in range(scifi_start, scifi_end):
            edge_list.append([i, i+1]) #edges in scifi segment

        #edges between UT and SciFi
        edge_list.append([ut_end, scifi_start])


    track_tensor=torch.tensor(node_ft, dtype=torch.float)
    edge_id=torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    #one element tensor (label true 1/ flase 0)
    label=torch.tensor([label], dtype=torch.long)

    return Data(x=track_tensor, edge_index=edge_id, y=label)


#plot graph xz projection
def plot_graph_xz(graph, name, track_id):
    x=graph.x
    edge_index=graph.edge_index
    z=x[:, 2].numpy()
    x_x=x[:, 0].numpy()
    det_id=x[:,-1].numpy()

    ut_mask=det_id==0 #only UT hits
    scifi_mask=det_id==1 #only SciFi hits
    #nodes
    plt.scatter(z[ut_mask], x_x[ut_mask], label="UT")
    plt.scatter(z[scifi_mask], x_x[scifi_mask], label="SciFi")
    for src, tgt in edge_index.t().numpy():
        plt.plot([z[src], z[tgt]], [x_x[src], x_x[tgt]])
#ENG
    # plt.title(f"Graph {track_id} in xz plane")

#PL
    plt.xlabel("z [mm]")
    plt.ylabel("x [mm]")
    plt.title(f"Graf {track_id} w projekcji zx")
    plt.grid(True, alpha=0.3)
    plt.margins(x=0.05, y=0.1)
    plt.tight_layout()
    os.makedirs("tracks_graphs_xz", exist_ok=True)
    filename=os.path.join("tracks_graphs_xz", name)
    plt.savefig(filename, dpi=150)
    plt.close()
    logging.info(f"Saved {filename}")

#plot graph yz projection
def plot_graph_yz(graph, name, track_id):
    x=graph.x
    edge_index=graph.edge_index
    z=x[:, 2].numpy()
    y=x[:, 1].numpy()
    det_id=x[:,-1].numpy()

    ut_mask=det_id==0
    scifi_mask=det_id ==1

    #nodes
    plt.figure()
    plt.scatter(z[ut_mask], y[ut_mask], label="UT")
    plt.scatter(z[scifi_mask], y[scifi_mask], label="SciFi")
    for src, tgt in edge_index.t().numpy():
        plt.plot([z[src], z[tgt]], [y[src], y[tgt]])
#ENG
    # plt.title(f"Graph {track_id} in yz plane")
#PL
    plt.xlabel("z [mm]")
    plt.ylabel("y [mm]")
    plt.title(f"Graf {track_id} w projekcji zy")
    plt.grid(True, alpha=0.3)
    plt.margins(x=0.05, y=0.1)
    plt.tight_layout()
    os.makedirs("tracks_graphs_yz", exist_ok=True)
    filename=os.path.join("tracks_graphs_yz", name)
    plt.savefig(filename, dpi=150)
    plt.show()
    logging.info(f"Saved {filename}")

if __name__=='__main__':
    events =load_downstream_tracks("sample_small_new_data.csv")
    events=[track for track in events if track["isDownstreamTrack"]]
    events =calc_avg_ut_xy_position(events)

    i=1
    ut=events[i]
    false_id=random.sample([f_id for f_id in range(len(events)) if f_id!=i],8)
    scifi_tracks=[events[i]] +[events[f_id] for f_id in false_id]

    graph = build_graph_from_track(ut, scifi_tracks, label=1)
    print(graph)
    plot_graph_yz(graph,f"track_{i:02d}.png",track_id=ut["track_id"])



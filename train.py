import logging, torch, random, time
from read_data import load_downstream_tracks, calc_avg_ut_xy_position
from load_data import TrackData
from torch_geometric.loader import DataLoader
from model_pm import TrackMessPassMod
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#readable time format instead of times in seconds
def format_time(full_t_seconds):
    t_hours=int(full_t_seconds//3600)
    t_minutes=int((full_t_seconds%3600)//60)
    t_seconds=full_t_seconds%60
    return f"{t_hours:02d}:{t_minutes:02d}:{t_seconds:06.3f}"

start_time=time.time()

logging.basicConfig(level=logging.INFO, format='%(message)s')

logging.info("\n------------------------------------------------------------------------------------")
logging.info("----------------------------- TRACK RECONSTRUCTION ---------------------------------\n")

logging.info("\n----------------------------- STARTED DATA LOADING ---------------------------------\n")
load_start_t=time.time()
events = load_downstream_tracks("complete_info_downstream_data.csv")
events = calc_avg_ut_xy_position(events)
random.shuffle(events)

n=len(events)
logging.info(f"Loaded {n} tracks.")
events=[track for track in events if track["isDownstreamTrack"]]
n=len(events)
n_train=int(0.8*n) #80% for training, 20 % for testing
train_events=events[:n_train]
test_events=events[n_train:]
logging.info(f"Dropped false tracks from data. Remained {len(train_events)} training tracks and {len(test_events)} test tracks.")
load_end_t=time.time()
load_time=format_time(load_end_t-load_start_t)
logging.info(f"Load stage runtime: {load_time}")
logging.info("\n------------------------------ ENDED DATA LOADING ----------------------------------\n")

logging.info("---------------------------- STARTED GRAPH BUILDING --------------------------------\n")
graph_b_start_t=time.time()
train_dataset=TrackData(train_events, n_false=4)
test_dataset=TrackData(test_events, n_false=100)
loader = DataLoader(train_dataset, batch_size=16)
test_loader=DataLoader(test_dataset, batch_size=16)
graph_b_end_t=time.time()
graph_b_time=format_time(graph_b_end_t-graph_b_start_t)
logging.info(f"Graph building stage runtime: {graph_b_time}")
logging.info("----------------------------- ENDED GRAPH BUILDING ---------------------------------\n")

hit_chr=6
neurons=32

labels = torch.cat([data.y for data in train_dataset]).view(-1)

model =TrackMessPassMod(hit_chr, neurons)
#criterion: loss function - how much wrong is model prediction
#BCE = Binary Cross Entropy WLL (because we have binary problem (true/false) and BCEWLL operates on logits (has sigmoid inside))
criterion=nn.BCEWithLogitsLoss()

#optimizer: optimizing weight of networks
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)

#training
logging.info("\n------------------------------- STARTED TRAINING -----------------------------------\n")
training_start_t=time.time()
epochs=40
for epoch in range(epochs):
    epoch_start_t=time.time()
    model.train()
    total_loss=0.0
    for batch in loader:
        optimizer.zero_grad()
        logits=model(batch)
        labels=batch.y.view(-1).float()
        # logging.info(f"\nlogits: {logits[:].detach()}")
        # logging.info(f"labels: {labels[:]}\n")
        loss=criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*labels.size(0)
    avg_loss=total_loss/len(train_dataset)
    epoch_end_t=time.time()
    epoch_time=epoch_end_t-epoch_start_t
    logging.info(f"----------------------------------------------\
                 \nEpoch {epoch:02d} | loss={avg_loss:.4f} | time={format_time(epoch_time)}")
training_end_t=time.time()
training_time=format_time(training_end_t-training_start_t)
logging.info("----------------------------- TRAINING ENDED ---------------------------------\n")
logging.info(f"Train stage runtime: {training_time}")

logging.info("----------------------------- STARTED TESTING ---------------------------------\n")
test_start_t=time.time()
model.eval()

#no grad - no training
with torch.no_grad():
    all_probs=[] #all tracks
    all_labels=[]
    all_ut_ids=[]

    for batch in test_loader:
        logits = model(batch) #logit: number at the end of model (-inf, inf)
        probs = torch.sigmoid(logits) #change logit to probability (0, 1)
        labels = batch.y.view(-1)

        all_probs.append(probs)
        all_labels.append(labels)
        all_ut_ids.append(batch.ut_id)

all_ut_ids=torch.cat(all_ut_ids)
all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)
probs_downstream=all_probs[all_labels==1].numpy()
probs_ghost=all_probs[all_labels==0].numpy()
test_end_t=time.time()
test_time=format_time(test_end_t-test_start_t)
logging.info("\n-------------------------------- TESTING ENDED -------------------------------------\n")
logging.info(f"Test stage runtime: {test_time}")

true_match=0
all=0

#ENG
# plt.figure()
# plt.hist(probs_downstream, bins=30, label="Downstream tracks",alpha=0.6, density=True)
# plt.hist(probs_ghost, bins=30, label="Ghosts",alpha=0.6, density=True)
# plt.xlabel("Predicted probability")
# plt.ylabel("Density")
# plt.title("Probability distribution")
# plt.legend()
# plt.show()

#PL
plt.figure(figsize=(7, 5))
plt.hist(probs_downstream, bins=30, label="Tory prawdziwe",alpha=0.8, density=True, color="cornflowerblue", linewidth=1.5)
plt.hist(probs_ghost, bins=30, label="Tory fałszywe",alpha=0.5, density=True, color="orangered",edgecolor="orangered",hatch="///", linewidth=1)
plt.xlabel("Przewidywane prawdopodobieństwo", fontsize=11)
plt.ylabel("Gęstość prawdopodobieństwa", fontsize=11)
plt.title("Rozkład prawdopodobieństwa", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

#PL
plt.figure(figsize=(7, 5))
plt.hist(probs_downstream, bins=30, label="Tory prawdziwe",alpha=0.8, density=True, color="cornflowerblue", linewidth=1.5)
plt.hist(probs_ghost, bins=30, label="Tory fałszywe",alpha=0.5, density=True, color="orangered",edgecolor="orangered",hatch="///", linewidth=1)
plt.xlabel("Przewidywane prawdopodobieństwo", fontsize=11)
plt.yscale("log")
plt.ylabel("Gęstość prawdopodobieństwa (skala logarytmiczna)", fontsize=11)
plt.title("Rozkład prawdopodobieństwa", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()


best5_match_id=0
true_match5=0

ks=[1,2,3,5, 10]
accs=[]

for k in ks:
    correct=0
    total=0
    for ut in torch.unique(all_ut_ids):
        mask=all_ut_ids==ut #only ut hits
        probs_ut =all_probs[mask]
        labels_ut =all_labels[mask]

        topk=torch.topk(probs_ut, k=min(k, len(probs_ut))).indices
        correct+= (labels_ut[topk]== 1).any().item()
        total+=1

    accs.append(correct / total)

logging.info("\nSkuteczność Top-k:")
for k, acc in zip(ks, accs):
    logging.info(f"  Top-{k}: {acc:.3f}")

#ENG
# plt.plot(ks, accs, marker="o")
# plt.xlabel("k")
# plt.ylabel("Top-k accuracy")
# plt.title("Tracking efficiency vs number of kept candidates")
# plt.grid(True)
# plt.show()

#PL
plt.figure(figsize=(7, 5))
plt.plot(ks, accs, marker="o", linewidth=2, color="cornflowerblue")
plt.xlabel("Liczba rozważanych kandydatów", fontsize=11)
plt.ylabel("Skuteczność", fontsize=11)
plt.title("Skuteczność rekonstrukcji w funkcji liczby kandydatów", fontsize=12)
plt.xticks(ks, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


logging.info("\n------------------------------------------------------------------------------------")
end_time=time.time()
runtime=format_time(end_time-start_time)
logging.info(f"Total program runtime: {runtime}")





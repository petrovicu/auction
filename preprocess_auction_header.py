import csv

with open("E:/datasets/pref/auction.csv") as fr, open("E:/datasets/pref/pref_auction.csv", "w", newline='') as fw:
    cr = csv.reader(fr)
    cw = csv.writer(fw)
    cw.writerow(["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "call"])
    cw.writerows(cr)


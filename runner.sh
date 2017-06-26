mkdir -p output_100_notimes/dists
mkdir -p output_10_notimes/dists
mkdir -p output_1_notimes/dists
# parallel 10
python3 run.py configs/config_1_notimes.json
python3 run.py configs/config_10_notimes.json
python3 run.py configs/config_100_notimes.json
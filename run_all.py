import subprocess
import json
import argparse

jobs = [
    {
        "epsilon_frames": 10 ** 6,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "fsa": True,
        "machine": "ngcv8",
        "replay_initial": 50000,
        "video_interval": 1000000,
        "frame_stop": 3010000,
        "dqn_model": "FSADQNScaling"
    },
    {
        "epsilon_frames": 10 ** 6,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00005,
        "gamma": 0.99,
        "fsa": True,
        "machine": "ngcv8",
        "replay_initial": 50000,
        "video_interval": 1000000,
        "frame_stop": 3010000,
        "dqn_model": "FSADQNScaling"
    },
    {
        "epsilon_frames": 10 ** 6,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.000025,
        "gamma": 0.99,
        "fsa": True,
        "machine": "ngcv8",
        "replay_initial": 50000,
        "video_interval": 1000000,
        "frame_stop": 3010000,
        "dqn_model": "FSADQNScaling"
    },
#   {
#         "epsilon_frames": 10 ** 6,
#         "epsilon_start": 1.0,
#         "epsilon_final": 0.1,
#         "learning_rate": 0.0001,
#         "gamma": 0.99,
#         "fsa": True,
#         "machine": "ngcv8",
#         "replay_initial": 50000,
#         "video_interval": 1000000,
#         "frame_stop": 3010000,
#         "dqn_model": "FSADQNAffine"
#     },
#     {
#         "epsilon_frames": 10 ** 6,
#         "epsilon_start": 1.0,
#         "epsilon_final": 0.1,
#         "learning_rate": 0.00005,
#         "gamma": 0.99,
#         "fsa": True,
#         "machine": "ngcv8",
#         "replay_initial": 50000,
#         "video_interval": 1000000,
#         "frame_stop": 3010000,
#         "dqn_model": "FSADQNAffine"
#     },
#     {
#         "epsilon_frames": 10 ** 6,
#         "epsilon_start": 1.0,
#         "epsilon_final": 0.1,
#         "learning_rate": 0.000025,
#         "gamma": 0.99,
#         "fsa": True,
#         "machine": "ngcv8",
#         "replay_initial": 50000,
#         "video_interval": 1000000,
#         "frame_stop": 3010000,
#         "dqn_model": "FSADQNAffine"
#     },
    {
      "epsilon_frames": 10 ** 6,
      "epsilon_start": 1.0,
      "epsilon_final": 0.1,
      "learning_rate": 0.000025,
      "gamma": 0.99,
      "fsa": True,
      "machine": "ngcv8",
      "replay_initial": 50000,
      "video_interval": 1000000,
      "frame_stop": 3010000,
      "dqn_model": "FSADQNParallel"
    }
] # list of dictionaries (json)

cloud = []
local = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=False, action="store_true", help="Enable verbose output")
    parser.add_argument("--file", default='', help="Input file")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            jobs = json.loads(open(args.file, "r").read())

    for job in jobs:
        if "machine" not in job:
            print("Machine not specified in job: ", job)
        elif job["machine"] == "local":
            local.append(job)
        else:
            cloud.append(job)

    with open("cloud.json", "w") as f:
        f.write(json.dumps(cloud))
    with open("local.json", "w") as f:
        f.write(json.dumps(local))

    if args.v:
        subprocess.call("python run_local.py -v --file local.json & "
                        "python run_sequential.py -v -p --file cloud.json", shell=True)
    else:
        subprocess.call("python run_local.py --file local.json & "
                        "python run_sequential.py -p --file cloud.json", shell=True)

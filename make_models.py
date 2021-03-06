import itertools
import argparse
import json
import subprocess

settings = {
    "epsilon_frames": [10 ** 6, 10 ** 6 / 2, 10 ** 6 * 2],
     "epsilon_start": [1.0],
     "epsilon_final": [0.1],
     "learning_rate": [0.00003, 0.000025],
             "gamma": [0.99],
         "dqn_model": ["FSADQNScaling"],
               "fsa": [True],
           "machine": ["ngcv4"],
    "replay_initial": [500],
    "video_interval": [1000],
        "frame_stop": [3000]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", default=1, type=int, help="Number of repeat jobs")
    parser.add_argument("--file", default='', help="Output file")
    parser.add_argument("--run", default=False, action="store_true", help="Run jobs")
    parser.add_argument("-v", default=False, action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    if len(args.file) == 0:
        args.file = "jobs.json"

    # Makes all the permutations of the above settings
    keys, values = zip(*settings.items())
    jobs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(len(jobs), "jobs created")
    print("Writing each job", args.reps, "time(s) to", args.file)

    # Repeats each job specified number of times
    output = []
    for job in jobs:
        for i in range(args.reps):
            output.append(job)

    # Writes jobs to json file
    with open(args.file, "w") as f:
        f.write(json.dumps(output))

    if args.v:
        print(output)

    if args.run:
        # Runs jobs
        if args.v:
            subprocess.call("python run_all.py -v --file " + args.file, shell=True)
        else:
            subprocess.call("python run_all.py --file " + args.file, shell=True)

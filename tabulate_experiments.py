import os
import matplotlib.pyplot as plt
import csv
import yaml
import pickle

def add_experiment(job_id, results_path):
    outfile = "/output.txt"
    paramfile = "/params.txt"
    modelfile = "/model/data.pkl"
    tabulatedfile = "/tabulated_experiments.csv"

    file_exists = True
    if not os.path.exists(results_path + tabulatedfile):
        file_exists = False

    skip_cols = ['env_name', 'stop_reward', 'run_name', 'replay_size', 'replay_initial', 'target_net_sync',
                'gamma', 'batch_size', 'video_interval', 'fsa', 'plot', 'telemetry']

    column_names = ['job_id']
    column_entries = [str(job_id)]
    with open(results_path + "/" + job_id + paramfile) as f:
        for line in f:
            entries = line.rstrip('\n').split(": ")
            if entries[0] in skip_cols:
                continue
            column_names.append(entries[0])
            column_entries.append(entries[1])
    column_names = ','.join(column_names) + '\n'
    column_entries = ','.join(column_entries) + '\n'

    with open(results_path + tabulatedfile, 'a') as f:
        if not file_exists:
            f.write(column_names)
        f.write(column_entries)

def tabulate_experiments():
    curdir = os.path.abspath(__file__)
    results_path = os.path.abspath(os.path.join(curdir, '../results/'))
    tabulatedfile = "/tabulated_experiments.csv"
    if os.path.exists(results_path + tabulatedfile):
        os.remove(results_path + tabulatedfile)
    job_ids = os.listdir(results_path)
    job_ids.sort(key=int)
    for job_id in job_ids:
        add_experiment(job_id, results_path)

def visualize_data(job_list):
    print("Plotting results for jobs {}".format(job_list))
    curdir = os.path.abspath(__file__)
    outfile = "/output.txt"
    paramfile = "/params.txt"
    modelfile = "/model/data.pkl"

    plt.figure(1)
    results = {}
    fieldnames = ['frames', 'games', 'mean reward', 'mean score', 'max score']
    for job_id in job_list:
        results[job_id] = {}
        for field in fieldnames:
            results[job_id][field] = []
        results_path = os.path.abspath(os.path.join(curdir, '../results/' + str(job_id)))
        with open(results_path + outfile) as f:
            reader = csv.DictReader(f)
            for line in reader:
                for field in fieldnames:
                        results[job_id][field].append(float(line[field]))

        model_path = results_path + modelfile
        model_info = pickle.load(open(model_path, "rb"))
        model_name = model_info[2]
        tm_acc = model_info[1]
        ave_score = model_info[0]
        print("{} | {} | TM acc: {} | Ave score: {}".format(job_id, model_name, tm_acc, ave_score))

        params_path = results_path + paramfile
        param_dict = yaml.load(open(params_path))

        i = 0
        for key in results[job_id].keys():
            if key != 'games' and key != 'frames':
                num_keys = str(len(results[job_id].keys()) - 2)
                subplot_num = int(num_keys + '1' + str(i + 1))
                # print(subplot_num)
                plt.subplot(subplot_num)
                plt.ylabel(key)
                plt.xlabel('games')
                plt.plot(results[job_id]['games'], results[job_id][key], label="{} | {}: e frames: {:9} | lr: {:10.7}".format(
                    job_id, model_name, int(param_dict['epsilon_frames']), format(float(param_dict['learning_rate']), 'f')))
                i += 1
    plt.legend()
    plt.show()



if __name__ == "__main__":
    tabulate_experiments()
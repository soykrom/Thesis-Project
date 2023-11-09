import pickle
import subprocess

with open('common/coefficients.pkl', 'rb') as file:
    co_list_pl = pickle.load(file)
    co_list_dist = pickle.load(file)
    co_list_done = pickle.load(file)

for co_pl, co_dist, co_done in zip(co_list_pl, co_list_dist, co_list_done):
    print(co_pl)
    print(co_dist)
    print(co_done)
    command = f'C:\\IST\\Tese\\Thesis-Project\\pytorch-soft-actor-critic\\main.py \
    --coefficients {co_pl} {co_dist} {co_done}'

    print(command)
    # Use subprocess.PIPE for stdout to capture the output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)

    # Print output as the command runs
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # Get the final output after the command has completed
    final_output, _ = process.communicate()

    print("Final Output:", final_output.strip())

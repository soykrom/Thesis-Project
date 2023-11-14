import pickle
import subprocess

with open('common/coefficients.pkl', 'rb') as file:
    co_list_pl = pickle.load(file)
    co_list_dist = pickle.load(file)
    co_list_done = pickle.load(file)

count = 0
reward_list = []
for co_pl, co_dist, co_done in zip(co_list_pl, co_list_dist, co_list_done):
    count += 1
    print(co_pl)
    print(co_dist)
    print(co_done)
    command = f'C:\\IST\\Tese\\Thesis-Project\\pytorch-soft-actor-critic\\main.py \
    --coefficients {co_pl} {co_dist} {co_done}'

    print(command + '\n' + "Count: ", count)
    # Run process
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)

    # Prints output as the command runs
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # Final reward
    avg_reward, _ = process.communicate()

    reward_list.append([[co_pl, co_dist, co_done], avg_reward])

    print("Reward:", avg_reward.strip())

    with open('common/rewards.pkl', 'ab') as filename:
        pickle.dump(reward_list, filename)

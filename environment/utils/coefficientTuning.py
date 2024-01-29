import pickle
import subprocess

with open('common/coefficients.pkl', 'rb') as file:
    co_list = pickle.load(file)

count = 0
reward_list = []
for coefficient in reversed(co_list):
    count += 1
    print(coefficient[0])
    print(coefficient[1])
    print(coefficient[2])

    command = f'D:\\IST\\Tese\\Thesis-Project\\pytorch-soft-actor-critic\\main.py \
    --coefficients {coefficient[0]} {coefficient[1]} {coefficient[2]}'

    print(command + '\n' + "Count: ", count)
    # Run process
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)

    output = []
    # Prints output as the command runs
    while True:
        # Read line from standard output
        output_line = process.stdout.readline()
        if output_line == '' and process.poll() is not None:
            break
        if output_line:
            print(output_line.strip())
            output.append(output_line.strip())

    # Final reward
    avg_reward = output[-1]
    print("Reward:", avg_reward)

    reward_list.append([coefficient, avg_reward])

    with open('common/rewards.pkl', 'wb') as filename:
        pickle.dump(reward_list, filename)


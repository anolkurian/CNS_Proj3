import subprocess

results = []
max = 0
maxA = [[]]
cmdAns = ''
with open('results.txt', 'a') as f:
    for num_epoch in range(100, 1001, 100):
        for num_hidden in [32, 64, 128, 256, 512, 1024]:
            for l2_reg_const in [0.0, 0.001, 0.005, 0.01]:
                for num_shadows in range(10, 201, 10):
                    nn_desc = "simple,{},{}".format(num_hidden, l2_reg_const)
                    cmd = "python3 hw.py problem4 {} {} {}".format(nn_desc, num_epoch, num_shadows)
                    # execute the command and capture the output
                    output = subprocess.check_output(cmd.split())
                    output_lines = output.decode().strip().split('\n')
                    print(output_lines)
                    # write the output lines to the file
    #                 f.write(f"Command: {cmd}\n")
    #                 for line in output_lines:
    #                     f.write(f"{line}\n")
    #                 f.write('\n')
    # f.write(f"FINAL ANS\n")
    # f.write(f"{cmdAns}\n")
    # f.write(f"{maxA}\n")
    # f.write('\n')

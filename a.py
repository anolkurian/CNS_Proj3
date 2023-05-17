import subprocess

results = []
max = 0
maxA = [[]]
cmdAns = ''
with open('results.txt', 'a') as f:
    for num_epoch in range(100, 1001, 100):
        for num_hidden in [32, 64, 128, 256, 512, 1024]:
            for l2_reg_const in [0.0, 0.001, 0.005, 0.01]:
                nn_desc = "simple,{},{}".format(num_hidden, l2_reg_const)
                cmd = "python3 hw.py problem1 {} {}".format(nn_desc, num_epoch)
                # execute the command and capture the output
                output = subprocess.check_output(cmd.split())
                output_lines = output.decode().strip().split('\n')
                nums = output_lines[0].split(',')
                ttAccRatio = float(nums[0])/float(nums[2])
                if(ttAccRatio > max):
                    max = ttAccRatio
                    maxA[0] = nums
                    cmdAns = cmd
                print(maxA)
                # write the output lines to the file
                f.write(f"Command: {cmd}\n")
                for line in output_lines:
                    f.write(f"{line}\n")
                f.write('\n')
    f.write(f"FINAL ANS\n")
    f.write(f"{cmdAns}\n")
    f.write(f"{maxA}\n")
    f.write('\n')

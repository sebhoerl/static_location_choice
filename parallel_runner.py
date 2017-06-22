import subprocess
import sys
import re

with open(sys.argv[1]) as f:
    commands = list(f) + ["__last__"]

parallel_limit = 1
processes = {}
barrier = False

for i, command in enumerate(commands):
    command = command.strip()
    barrier = False

    if command.startswith("#"):
        annotation = command[1:].strip()
        understood = False

        if annotation.startswith("parallel"):
            match = re.search(r"parallel\s+([0-9]+)", annotation)

            if match:
                parallel_limit, before = int(match.group(1)), parallel_limit
                print("Setting number of parallel tasks from %d to %d" % (before, parallel_limit))
                understood = True

        if annotation.startswith("barrier"):
            print("Barrier, waiting for %d tasks" % len(processes))
            barrier = True
            understood = True

        if not understood:
            print("Did not understood annotation in line %d: %s" % (i, annotation))

    else:
        while len(processes) >= parallel_limit or (barrier and len(processes) > 0) or (command == "__last__" and len(processes) > 0):
            remove = []

            for number, process in processes.items():
                code = process.poll()

                if code is not None:
                    if code != 0: print("Exit % 4d) Return code %d" % (number, code))
                    remove.append(number)

            for number in remove: del processes[number]

        if command != "__last__":
            print("Running % 4d) %s" % (i, command))
            processes[i] = subprocess.Popen(command, shell = True)

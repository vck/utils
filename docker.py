# stop and remove running container

import os
import fire

def remove_running_containers():
    os.system("docker ps -a >> running-containers.txt")

    with open("running-containers.txt") as running:
        for line in running:
            # check for header
            len_header = 8

            if len(line.strip('\n').split()) == len_header:
                continue
            
            line = line.strip('\n').split()
            
            if "Exited" not in line:
                continue

            exited_container = line[0]
            
            # stop and delete container
            stop_status = os.system(f"docker stop {exited_container}")
            
            stop_previx = "successfull" if stop_status == 0 else "unsuccesfull"
            print(f"{exited_container} {stop_previx}ly stopped")

            delete_status = os.system(f"docker rm {exited_container}")
            delete_previx = "successfull" if delete_status == 0 else "unsuccesfull"
            print(f"{exited_container} {stop_previx}ly deleted")

if __name__ == "__main__":
    fire.Fire()

import random

from uav_env import UAVenv

new_env = UAVenv()

for i in range(5):
    print(new_env.step([random.randint(1,5),random.randint(1,5),random.randint(1,5),random.randint(1,5),random.randint(1,5)]))




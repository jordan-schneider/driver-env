This is an implementaiton of the driver environment first introduced  in "Active Preference-Based Learning of Reward Functions" by Sadigh et. al. I provide both the orignal trajectory based implementation and a gym environment more suitable for training RL algorithms. This repository also provides type hints.

To access the gym environment, import the package and make 'driver-v1' while providing a reward and time horizon.
```python
import driver
import gym
import numpy as np
reward = np.ones(4,)
env = gym.make("driver-v1", reward=reward, horizon=50)
```

To access the trajectory based environment, construct a driver object
```python
from driver.model import Driver
# Make reward vector, actions
env = Driver()
env.feed(actions)
features = env.get_features()
epsiode_return = reward @ features
```


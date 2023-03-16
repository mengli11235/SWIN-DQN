# Deep Reinforcemnet Learning with Swin Transformer

This repo is the implementation of the paper 'Deep Reinforcemnet Learning with Swin Transformer' on ALE games. 

# Requirements

atari-py installed from https://github.com/kastnerkyle/atari-py  
torch='1.0.1.post2'  
cv2='4.0.0'  
matplotlib='3.3.2'
timm='0.5.4'
scipy='1.7.3'
imageio='2.9.0'

# Usage
```run.py``` is the main file.
```visual_plot.py``` is the file to plot eval results.
```plot_kernel.py``` is the file to visualize models.

To run the experiment, set ```info``` in ```run.py```. To run different games, set game names in ```info['GAME']```. Set ```info['IMPROVEMENT']``` to use Swin Transformer or not. To run ```visual_plot.py``` and ```plot_kernel.py```, the correct model path needs to be set.

```bash
python run.py
```

# Credit

Credit to which the project is built on: 

https://github.com/johannah/bootstrap_dqn

https://github.com/microsoft/Swin-Transformer

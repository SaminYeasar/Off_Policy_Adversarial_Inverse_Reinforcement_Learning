# Off Policy Adversarial Inverse Reinforcement Learning (off-policy-AIRL)

Source code to accompany [Off Policy Adversarial Inverse Reinforcement Learning](https://openreview.net/forum?id=9mp5d073IhX).

If you use this code for your research, please consider citing the paper:

```
@article{arnob2020off,
  title={Off-Policy Adversarial Inverse Reinforcement Learning},
  author={Arnob, Samin Yeasar},
  journal={arXiv preprint arXiv:2005.01138},
  year={2020}
}
```




# To run Custom environments


#### Folder required  
    *`inverse_rl (from :https://github.com/justinjfu/inverse_rl)
    * rllab
    * sandbox`

#### Library requires
    * rllab (https://github.com/openai/rllab)
    * PyTorch
    * Python 2 
    * mjpro131 
    * pip install mujoco-py==0.5.7 


---

# To run MuJoCo environments

#### Library requires
    * PyTorch
    * Python 3
    * mujoco-py==1.50.1.68 


---
# Download saved data
* [Expert trajectory](https://drive.google.com/drive/folders/1pqc9nhUdKFymc0hFUj2bL8KB9Mq9lj-p?usp=sharing)

---
## Compute Imitation performance:



```
python Train.py --seed 0 \
                --env_name "HalfCheetah-v2" \
                --learn_temperature \
                --policy_name "SAC"
```
Description of different arguments are following:

* **Enviroment options**: `
    * OpenAI gym: `HalfCheetah-v2, Ant-v2, Hopper-v2, Walker2d-v2` 
    * Custom environments `CustomAnt-v0, PointMazeLeft-v0`
* **learn_temperature**:
    * allows the temperature parameter of SAC to be a learning parameter
*    **Policy options** `SAC`, `SAC_MCP`(k=8 premitive policies), `SAC_MCP2` (k=4 premitive policies)


![](https://imgur.com/ljK0XeB.png)
![](https://imgur.com/BA9jff9.png)

---
## Compute Transfer Learning:

##### Transfer learning experiment is computed on Custom environment from (https://github.com/justinjfu/inverse_rl/tree/master/inverse_rl)
```
python ReTrain.py --seed 0 
                --env_name "DisabledAnt-v0" \
                --learn_temperature \
                --policy_name "SAC" \
                --initial_state "random"  \
                --initial_runs "policy_sample"\
                --load_gating_func\
                --learn_actor 
```

Description of different arguments are following:

* **Enviroment options**: `
    * Custom environments `DisabledAnt-v0, PointMazeRight-v0`

* **learn_temperature**:
    * allows the temperature parameter of SAC to be a learning parameter
    
*    **Policy options** `SAC`, `SAC_MCP`(k=8 premitive policies), `SAC_MCP2` (k=4 premitive policies)
*    **initial_state** 
        * `zero` environment starts from same state
        * `random` environment starts from random states

                --initial_runs "policy_sample"\
*    **load_gating_func**
        *    applicable only for `SAC_MCP` and `SAC_MCP2`
        *    if flagged, loads `gating function` from imitation training
        *    if not flagged, random initialization of the `gating function`
*    **learn_actor** 
        *    applicable only for `SAC_MCP` and `SAC_MCP2`
        *   if flagged, retrains `policy` and `gating function`
        *   if not flagged, retrain only `gating function`


```

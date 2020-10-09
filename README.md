# Off_Policy_AIRL


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
## Compute Imitation performance:


##### Transfer learning experiment is computed on Custom environment from (https://github.com/justinjfu/inverse_rl/tree/master/inverse_rl)
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




---
## Compute Transfer Learning:


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



### Results:

```


Plot.ipynb : iPython Notebook that plots the figures discussed in Paper


Results |- AIRL |- SAC - |-learn_temp_true
                |        |- learn_temp_false
                |         
                |- SAC_MCP |-learn_temp_true
                |          |- learn_temp_false
                |             
                |- SAC_MCP2  |-learn_temp_true
                             |- learn_temp_false

        |- AIRL_Retrain |- SAC |-learn_temp_true
                        |      |- learn_temp_false
                        | 
                        |- SAC_MCP |-learn_temp_true
                        |          |- learn_temp_false
                        |     
                        |- SAC_MCP2  |-learn_temp_true
                                     |- learn_temp_false
                                     
        |- Expert Data |- Expert Trajectory: Collected expert trajcetory to be used in training imitation task
                       |- Training_data: Data collected during training SAC, TD3 using environment rewards
                       
        |- DAC:  Contains data from DAC implementation
    

```

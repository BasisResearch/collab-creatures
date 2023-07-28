The directory 'mx_refactored' is version 1 of the refactored multi-agent modeling sandbox. It should have the following files:

	simulation.py 
	agents.py
	environments.py
	test_simulation.py 

The goal is to convert existing modeling sandbox (multiagent_foraging_many_sims.py, multiagent_modeling_tutorial.py, TreeWorld.py) into a format that is more amenable to inference. 

This code is a work in progress. Run test_simulation.py to do the following:

* create a new environment with food statistics of a particular kind of spatiotemporal structure 
* create a simulation with that environment and an agent type
* output the locations of the birds and the rewards via dataframes

Limitations of this code
* only 3 types of agents (see how I have implemented the agents in agents.py)
* only 1 type of food statistic ("drop food once")
* no animations
* class structure not efficient and probably prone to errors 


TO DO 
1. Fix the time indexing - data in the last time step is not recorded in the data frames.
2. Implement agent types more flexibly/robustly (I currently assume 4 sources of reward).
3. Get animations to work  
4. Add more fields to the dataframe for each bird: 
	* total calories acquired 
	* total calories consumed
	* directness of path
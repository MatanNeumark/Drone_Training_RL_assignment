This repository contains the code for the assignment of the course AE4350 Bio-inspired Intelligence Learning for Aerospace Applications.

To run the training script:

a. You need two files: 'REINFORCE_learning_Quad_V2.py' which performs the training, and 'QuadModel.py' which is the environment.

b. You can change the hyperparameters alpha (learning rate) and gamma (discount factor) and the desired number of episodes.

c. You can also specify render mode as 'human' if you want to see an animation of the drone in the environment.

![image](https://github.com/user-attachments/assets/4cae1625-301a-4549-a820-b8fbacd86cb6)

'REINFORCE_learning_Quad_V2.py' includes some plots to visualize training results and performance.

For post processing and data analysis, you can use the script 'Quad_learning_results_analysis.py.
It has loops to run batches to reduce results uncertainty and plots to see the effect of hyperparameter variation.

The image below shows the overall structure of the code and how information flows between classes.

![REINFORCE_arcitecture drawio](https://github.com/user-attachments/assets/ff52d7bc-ce18-44d5-a122-37f7dc327231)


![environment](https://github.com/user-attachments/assets/a67d64d7-616d-40b7-90cb-bd06098dbe6f)

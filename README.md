This repository contains the code and results for the assignment of the course AE4350 Bio-inspired Intelligence Learning for Aerospace Applications.

The learning task is for a drone to take off, fly to and hover inside the target zone (red square)
In the report, you can read about the method and results. The video below showcases the performance after training for 10,000 episodes.


https://github.com/user-attachments/assets/845eb6c1-0c40-4eb0-ade2-be7c50855fe2


To run the training script:

a. You need two files: 'REINFORCE_learning_Quad_V2.py' which performs the training, and 'QuadModel.py' which is the environment.

b. You can change the hyperparameters alpha (learning rate) and gamma (discount factor) and the desired number of episodes.

c. You can also specify the drone's mass, diagonal motor distance and even earth's gravity! Render mode can be set to 'human' if you want to see an animation of the drone in the environment.

![image](https://github.com/user-attachments/assets/4cae1625-301a-4549-a820-b8fbacd86cb6)
![image](https://github.com/user-attachments/assets/9701ea76-3089-455c-8c4d-4a08a915c985)


'REINFORCE_learning_Quad_V2.py' includes some plots to visualize training results and performance.

For post processing and data analysis, you can use the script 'Quad_learning_results_analysis.py.
It has loops to run batches to reduce results uncertainty and plots to see the effect of hyperparameter variation.

The image below shows the overall structure of the code and how information flows between classes.

![REINFORCE_arcitecture drawio](https://github.com/user-attachments/assets/ff52d7bc-ce18-44d5-a122-37f7dc327231)


![environment](https://github.com/user-attachments/assets/a67d64d7-616d-40b7-90cb-bd06098dbe6f)

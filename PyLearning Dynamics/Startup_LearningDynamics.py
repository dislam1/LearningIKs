import os
import sys

def Startup_LearningDynamics():
    # Prepare all needed folders to run the learning simulator
    print('Setting up paths for Learning Dynamics ......')
    #home_path = os.getcwd() + os.sep  # Home directory
    home_path = 'C:\\Users\\CluClu\\Downloads\\Maggioni Python\\'
    print(f'  Adding necessary folders to {home_path}...')
    sys.path.append(home_path + 'SOD_Evolve/')  # code to run the dynamics for obtaining observation
    sys.path.append(home_path + 'SOD_Learn/')  # code to run the learning routines
    sys.path.append(home_path + 'SOD_Utils/')  # Utility routines shared by both simulators
    sys.path.extend([os.path.join(home_path, 'SOD_Examples'),  # code to generate the examples for specific dynamics
                     os.path.join(home_path, 'SOD_External')])  # external package(s)
    print('done.')

Startup_LearningDynamics()

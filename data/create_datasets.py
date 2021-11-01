
import numpy as np
import pandas as pd




def create_control_case_translation_split_dataset():
    # create rotation matrix
    theta = np.pi/3
    c,s = np.cos(theta), np.sin(theta)
    R = np.array([[c,-s],[s,c]])

    # Control class
    num_samples = 200
    mean = np.array([1,1])
    cov = np.diag([5,0.1])
    control = np.random.multivariate_normal(mean, cov, size=(num_samples,))
    
    # Case class
    num_samples = 200
    mean = np.array([1,2]) # notice the difference
    cov = np.diag([5,0.1])
    case = np.random.multivariate_normal(mean, cov, size=(num_samples,))
    
    # rotate
    control = R@control.T
    case = R@case.T


    control_df = pd.DataFrame(data = {
        'X': control[0],
        'Y': control[1],
        'Class': 'Control'
    })

    case_df = pd.DataFrame(data = {
        'X': case[0],
        'Y': case[1],
        'Class': 'case'
    })
    
    df = pd.concat([control_df, case_df], ignore_index=True)
    df.to_csv('case_control_translation_split.csv', index=False)  
    return 0

if __name__ == '__main__':
    create_control_case_translation_split_dataset()
    


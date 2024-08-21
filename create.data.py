import numpy as np
import pandas as pd

n_users = 100
user_data = {
    "Age": np.random.randint(25, 66, size=n_users),
    "Gender": np.random.randint(0, 2, size=n_users),
    "Country": np.random.randint(0, 5, size=n_users),
    "Operating_System": np.random.randint(0, 4, size=n_users),
    "Has_Used_macOS": np.random.randint(0, 2, size=n_users),
    "Owns_MacBook": np.random.randint(0, 2, size=n_users),
    
}


df_users = pd.DataFrame(user_data)


file_path_users = "new_data.csv"
df_users.to_csv(file_path_users, index=False)

file_path_users
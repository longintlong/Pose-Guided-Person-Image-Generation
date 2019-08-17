import os
import random
import pandas as pd


test_img_dir = "data/DF_img_pose/filted_up_test"
# samples = random.sample(os.listdir(test_img_dir), 50)
pair_lst_path = os.path.join("data","DF_img_pose","PoseFiltered","fashion-pairs-test.csv")
pair_df = pd.read_csv(pair_lst_path)
from_lst, to_lst = pair_df['from'].tolist(), pair_df['to'].tolist()
for img in os.listdir(test_img_dir):
    if img not in from_lst and img not in to_lst:
        os.remove(os.path.join(test_img_dir,img))
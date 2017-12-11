instructions:
    - place stage1 a3d, a3daps, aps files in "input/competition_data/{a3d,a3daps,aps}"
      respectively
    - place stage2 a3d, a3daps, aps files in "input/competition_data/stage2/{a3d,a3daps,aps}"
      respectively
    - run "python run.py private_test"
    - output files are "cache\get_final_answer_csv\122369\'private_test'\ans1.txt",
      "cache\get_final_answer_csv\122369\'private_test'\ans2.txt"
requirements:
    - those in requirements.txt, along with elastix v4.8 and tensorflow v1.4.0
    - tested on windows 10
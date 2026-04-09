# python debug/add_boxnoise.py -n 0.0  -f exps/std/env_cfgs/objTracking.yaml

# python exps/std/run.py -a SHAC -t 1 -c posObs_VaryDis_H96
# python exps/std/run.py -a diff_dreamer -t 1 -c posObs_VaryDis_H96

# python exps/std/run.py -a SHAC -t 0 -c posObs_VaryDis_H96 -w SHAC_posObs_VaryDis_H96_1.zip
# python exps/std/run.py -a diff_dreamer -t 0 -c posObs_VaryDis_H96 -w DiffDreamer_posObs_VaryDis_H96_1.zip

# python debug/add_boxnoise.py -n 0.4  -f exps/std/env_cfgs/objTracking.yaml

# python exps/std/run.py -a SHAC -t 1 -c posObs_VaryDis_H96_noise
# python exps/std/run.py -a diff_dreamer -t 1 -c posObs_VaryDis_H96_noise

# python exps/std/run.py -a SHAC -t 0 -c posObs_VaryDis_H96_noise -w SHAC_posObs_VaryDis_H96_noise_1.zip
# python exps/std/run.py -a diff_dreamer -t 0 -c posObs_VaryDis_H96_noise -w DiffDreamer_posObs_VaryDis_H96_noise_1.zip

# python debug/comment.py -f envs/ObjectTrackingEnv.py -l 130 --toggle
# python debug/comment.py -f envs/ObjectTrackingEnv.py -l 69 --toggle
# python debug/comment.py -f envs/ObjectTrackingEnv.py -l 70 --toggle

# python debug/add_boxnoise.py -n 0.0  -f exps/std/env_cfgs/objTracking.yaml

# python exps/std/run.py -a SHAC -t 1 -c velObs_VaryDis_H96
# python exps/std/run.py -a diff_dreamer -t 1 -c velObs_VaryDis_H96

# python exps/std/run.py -a SHAC -t 0 -c velObs_VaryDis_H96 -w SHAC_velObs_VaryDis_H96_1.zip
# python exps/std/run.py -a diff_dreamer -t 0 -c velObs_VaryDis_H96 -w DiffDreamer_velObs_VaryDis_H96_1.zip

# python debug/add_boxnoise.py -n 0.4  -f exps/std/env_cfgs/objTracking.yaml

# python exps/std/run.py -a SHAC -t 1 -c velObs_VaryDis_H96_noise
# python exps/std/run.py -a diff_dreamer -t 1 -c velObs_VaryDis_H96_noise

# python exps/std/run.py -a SHAC -t 0 -c velObs_VaryDis_H96_noise -w SHAC_velObs_VaryDis_H96_noise_1.zip
# python exps/std/run.py -a diff_dreamer -t 0 -c velObs_VaryDis_H96_noise -w DiffDreamer_velObs_VaryDis_H96_noise_1.zip

# agile test
python debug/add_boxnoise.py -n 0.0  -f exps/std/env_cfgs/objTracking.yaml

# agile pos obs, no noise
python debug/comment.py -f envs/ObjectTrackingEnv.py -l 130 --comment
python debug/comment.py -f envs/ObjectTrackingEnv.py -l 69 --comment
python debug/comment.py -f envs/ObjectTrackingEnv.py -l 70 --comment
python exps/std/run.py -a diff_dreamer -t 1 -c posObs_VaryDis_H96_changeDisToSelfV_agile
python exps/std/run.py -a diff_dreamer -t 0 -w DiffDreamer_posObs_VaryDis_H96_changeDisToSelfV_agile_1.zip

python debug/add_boxnoise.py -n 0.4  -f exps/std/env_cfgs/objTracking.yaml
python exps/std/run.py -a diff_dreamer -t 1 -c posObs_VaryDis_H96_changeDisToSelfV_agile_noise
python exps/std/run.py -a diff_dreamer -t 0 -w DiffDreamer_posObs_VaryDis_H96_changeDisToSelfV_agile_noise_1.zip

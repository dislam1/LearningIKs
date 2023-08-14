import numpy as np

def check_rhoLT_independence(sys_info, data):
    # This function is not provided, so it needs to be implemented separately.
    pass

def system_checks(sys_info, obs_info, learningOutput):
    if "ModelSelection" not in sys_info["name"]:
        the_diff = check_rhoLT_independence(sys_info, obs_info["rhoLT"])
        print(f"For {sys_info['name']}, the true joint distribution of (r, \\dot[r]) has:")
        for k1 in range(1, sys_info["K"] + 1):
            for k2 in range(1, sys_info["K"] + 1):
                print(f"  at ({k1}, {k2}), the l1 difference is: {the_diff['rhoLTA_diff'][k1-1, k2-1]:10.4e}.")
        if sys_info["has_xi"]:
            print("The true joint distribution of (r, \\xi) has:")
            for k1 in range(1, sys_info["K"] + 1):
                for k2 in range(1, sys_info["K"] + 1):
                    print(f"  at ({k1}, {k2}), the l1 difference is: {the_diff['rhoLTXi_diff'][k1-1, k2-1]:10.4e}.")

    print(f"For {sys_info['name']}, the empirical joint distribution of (r, \\dot[r]) has:")
    the_diff = []
    for idx in range(len(learningOutput)):
        the_diff.append(check_rhoLT_independence(sys_info, learningOutput[idx]["rhoLTemp"]))
    for k1 in range(1, sys_info["K"] + 1):
        for k2 in range(1, sys_info["K"] + 1):
            diff_values = [x["rhoLTA_diff"][k1-1, k2-1] for x in the_diff]
            mean_diff = np.mean(diff_values)
            std_diff = np.std(diff_values)
            print(f"  at ({k1}, {k2}), the l1 difference is: {mean_diff:10.4e}±{std_diff:10.4e}")
    if sys_info["has_xi"]:
        print("The empirical joint distribution of (r, \\xi) has:")
        for k1 in range(1, sys_info["K"] + 1):
            for k2 in range(1, sys_info["K"] + 1):
                diff_values = [x["rhoLTXi_diff"][k1-1, k2-1] for x in the_diff]
                mean_diff = np.mean(diff_values)
                std_diff = np.std(diff_values)
                print(f"  at ({k1}, {k2}), the l1 difference is: {mean_diff:10.4e}±{std_diff:10.4e}")
    print("done")

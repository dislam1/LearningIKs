def restructure_sys_info_for_larger_N(N_ratio, sys_info):
    sys_info1 = sys_info.copy()
    switch_name = sys_info1["name"]
    if switch_name in ['OpinionDynamicsCont', 'OpinionDynamicsDisc', 'LennardJonesDynamics', 'LennardJonesDynamicsTruncated']:
        sys_info1["N"] *= N_ratio
        sys_info1["type_info"] = [1] * sys_info1["N"]
        if switch_name in ['OpinionDynamicsCont', 'OpinionDynamicsDisc']:
            sys_info1["mu0"] = 'lambda r: OD_init_config(r["d"], r["N"], 1)'
        elif switch_name in ['LennardJonesDynamics', 'LennardJonesDynamicsTruncated']:
            sys_info1["mu0"] = 'lambda r : LJ_init_config([], r["d"], r["N"], 2)'
    elif switch_name in ['PredatorPrey1stOrderSplines', 'PredatorPrey1stOrder', 'PredatorPrey2ndOrder']:
        N = sys_info1["N"] * N_ratio
        N_predators = 1
        N_preys = N - N_predators
        sys_info1["N"] = N
        sys_info1["type_info"] = [1] * N_preys + [2] * N_predators
        if switch_name in ['PredatorPrey1stOrderSplines', 'PredatorPrey1stOrder']:
            sys_info1["mu0"] = 'lambda r : PS_init_config(r["N"], r["type_info"], 1)'
        elif switch_name == 'PredatorPrey2ndOrder':
            sys_info1["mu0"] = 'lambda r : PS_init_config(r["N"], r["type_info"], 2)'
            sys_info1["agent_mass"] = [1] * N
            sys_info1["Fv"] = 'lambda v, xi: PS_2nd_order_friction(v, [1, 1], 2, sys_info["type_info"])'
    elif switch_name == 'PhototaxisDynamics':
        sys_info1["N"] *= N_ratio
        sys_info1["type_info"] = [1] * sys_info1["N"]
        sys_info1["mu0"] = 'lambda r : OD_init_config(r["d"], r["N"], 1)'
        sys_info1["agent_mass"] = [1] * sys_info1["N"]
        sys_info1["kappa"] = sys_info1["N"]
        sys_info1["kappaXi"] = sys_info1["N"]
        sys_info1["mu0"] = 'lambda r : PT_init_condition(r["d"], r["N"], 100, 100, 0.001)'
    print("restructure_sys_info_for_larger_N: {}".format(sys_info['N']))
    return sys_info1

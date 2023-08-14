def set_sys_info(sys_info, phiE, phiA, phiXi):
    # re-package the structure
    new_sys_info = sys_info.copy()
    new_sys_info['phiE'] = phiE
    new_sys_info['phiA'] = phiA
    new_sys_info['phiXi'] = phiXi
    return new_sys_info

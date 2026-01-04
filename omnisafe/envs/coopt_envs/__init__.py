from safety_gymnasium.utils.registration import register  # register in safety_gymnasium

# v0, continuous-discrete, CMDP
register(id='COCHTCLT-v0', entry_point='omnisafe.envs.coopt_envs.safe_coopt:CHTCLTEMSEnv')
register(id='COWHVC-v0', entry_point='omnisafe.envs.coopt_envs.safe_coopt:WHVCEMSEnv')
register(id='COWHVC3000-v0', entry_point='omnisafe.envs.coopt_envs.safe_coopt:WHVC3000EMSEnv')
register(id='COHDUDDS-v0', entry_point='omnisafe.envs.coopt_envs.safe_coopt:HDUDDSEMSEnv')
register(id='COJE05-v0', entry_point='omnisafe.envs.coopt_envs.safe_coopt:JE05EMSEnv')
register(id='COHHDDT-v0', entry_point='omnisafe.envs.coopt_envs.safe_coopt:HHDDTEMSEnv')

# v999, continuous-discrete, CMDP, for mpc
register(id='COCHTCLT-v999', entry_point='omnisafe.envs.coopt_envs.safe_coopt_mpc:CHTCLTEMSEnv')
register(id='COWHVC-v999', entry_point='omnisafe.envs.coopt_envs.safe_coopt_mpc:WHVCEMSEnv')
register(id='COHDUDDS-v999', entry_point='omnisafe.envs.coopt_envs.safe_coopt_mpc:HDUDDSEMSEnv')
register(id='COJE05-v999', entry_point='omnisafe.envs.coopt_envs.safe_coopt_mpc:JE05EMSEnv')
register(id='COHHDDT-v999', entry_point='omnisafe.envs.coopt_envs.safe_coopt_mpc:HHDDTEMSEnv')

# v1, continuous, CMDP
register(id='COCHTCLT-v1', entry_point='omnisafe.envs.coopt_envs.safe_coopt_continuous:CHTCLTEMSEnv')
register(id='COWHVC-v1', entry_point='omnisafe.envs.coopt_envs.safe_coopt_continuous:WHVCEMSEnv')
register(id='COWHVC3000-v1', entry_point='omnisafe.envs.coopt_envs.safe_coopt_continuous:WHVC3000EMSEnv')
register(id='ACCWHVC-v1', entry_point='omnisafe.envs.coopt_envs.safe_acc_continuous:WHVCACCEnv')
register(id='ACCCHTCLT-v1', entry_point='omnisafe.envs.coopt_envs.safe_acc_continuous:CHTCLTACCEnv')
register(id='ACCWHVC3000-v1', entry_point='omnisafe.envs.coopt_envs.safe_acc_continuous:WHVC3000ACCEnv')
register(id='CutIn-v1', entry_point='omnisafe.envs.coopt_envs.safe_coopt_continuous:CutInEnv')
register(id='SeqEMSCHTCLT-v1', entry_point='omnisafe.envs.coopt_envs.safe_seq_ems_continuous:CHTCLTEMSEnv')
register(id='SeqACCCHTCLT-v1', entry_point='omnisafe.envs.coopt_envs.safe_seq_acc_continuous:CHTCLTEMSEnv')

# v2, continuous-discrete
register(id='COCHTCLT-v2', entry_point='omnisafe.envs.coopt_envs.coopt:CHTCLTEMSEnv')
register(id='COWHVC-v2', entry_point='omnisafe.envs.coopt_envs.coopt:WHVCEMSEnv')
register(id='COWHVC3000-v2', entry_point='omnisafe.envs.coopt_envs.coopt:WHVC3000EMSEnv')

# v3, continuous
register(id='COCHTCLT-v3', entry_point='omnisafe.envs.coopt_envs.coopt_continuous:CHTCLTEMSEnv')
register(id='COWHVC-v3', entry_point='omnisafe.envs.coopt_envs.coopt_continuous:WHVCEMSEnv')
register(id='COWHVC3000-v3', entry_point='omnisafe.envs.coopt_envs.coopt_continuous:WHVC3000EMSEnv')

# v4, continuous, CMDP, torque control
register(id='COCHTCLT-v4', entry_point='omnisafe.envs.coopt_envs.safe_co_torque_conti:CHTCLTEMSEnv')
register(id='COWHVC-v4', entry_point='omnisafe.envs.coopt_envs.safe_co_torque_conti:WHVCEMSEnv')
register(id='COWHVC3000-v4', entry_point='omnisafe.envs.coopt_envs.safe_co_torque_conti:WHVC3000EMSEnv')

# v5, continuous, torque control
register(id='COCHTCLT-v5', entry_point='omnisafe.envs.coopt_envs.co_torque_conti:CHTCLTEMSEnv')
register(id='COWHVC-v5', entry_point='omnisafe.envs.coopt_envs.co_torque_conti:WHVCEMSEnv')
register(id='COWHVC3000-v5', entry_point='omnisafe.envs.coopt_envs.co_torque_conti:WHVC3000EMSEnv')

# v6, continuous-discrete, CMDP, torque control
register(id='COCHTCLT-v6', entry_point='omnisafe.envs.coopt_envs.safe_co_torque:CHTCLTEMSEnv')
register(id='COWHVC-v6', entry_point='omnisafe.envs.coopt_envs.safe_co_torque:WHVCEMSEnv')
register(id='COWHVC3000-v6', entry_point='omnisafe.envs.coopt_envs.safe_co_torque:WHVC3000EMSEnv')
register(id='COHDUDDS-v6', entry_point='omnisafe.envs.coopt_envs.safe_co_torque:HDUDDSEMSEnv')

# v7, continuous-discrete, torque control
register(id='COCHTCLT-v7', entry_point='omnisafe.envs.coopt_envs.co_torque:CHTCLTEMSEnv')
register(id='COWHVC-v7', entry_point='omnisafe.envs.coopt_envs.co_torque:WHVCEMSEnv')
register(id='COWHVC3000-v7', entry_point='omnisafe.envs.coopt_envs.co_torque:WHVC3000EMSEnv')
register(id='COHDUDDS-v7', entry_point='omnisafe.envs.coopt_envs.co_torque:HDUDDSEMSEnv')
register(id='COJE05-v7', entry_point='omnisafe.envs.coopt_envs.co_torque:JE05EMSEnv')
register(id='COHHDDT-v7', entry_point='omnisafe.envs.coopt_envs.co_torque:HHDDTEMSEnv')

# v70, continuous-discrete, torque control, for mpc
register(id='COCHTCLT-v70', entry_point='omnisafe.envs.coopt_envs.co_torque_mpc:CHTCLTEMSEnv')
register(id='COWHVC-v70', entry_point='omnisafe.envs.coopt_envs.co_torque_mpc:WHVCEMSEnv')
register(id='COHDUDDS-v70', entry_point='omnisafe.envs.coopt_envs.co_torque_mpc:HDUDDSEMSEnv')
register(id='COJE05-v70', entry_point='omnisafe.envs.coopt_envs.co_torque_mpc:JE05EMSEnv')
register(id='COHHDDT-v70', entry_point='omnisafe.envs.coopt_envs.co_torque_mpc:HHDDTEMSEnv')


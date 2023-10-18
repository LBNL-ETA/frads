# How to set up a Three-Phase method workflow with a configuration file?

This guide will show you how to set up a Three-Phase method workflow with a configuration file. The configuration file is a ???? file that contains all the information needed to run a Three-Phase method workflow. 

**Workflow**

1. 


## 



cfg = fr.WorkflowConfig()
cfg.Model.scene = 
cfg.Model.materials =
cfg.Model.sensors =
cfg.Model.views =

rad_workflow = fr.ThreePhaseMethod(rad_cfg)
print(rad_workflow.mfile)
# Separate run to generate matrices and save to file
# A *.npz file will be generated in the current working directory
rad_workflow.config.settings.save_matrices = True
rad_workflow.generate_matrices(view_matrices=False)
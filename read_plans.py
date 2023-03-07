from ds_info.utils.io_utils import pkl_load


task = 'Task010_Colon'

plans = pkl_load(name="plans", path='/local/scratch/cgonzalez/Lifelong-nnUNet_storage/nnUNet_trained_models/nnUNet/3d_fullres/'+task+'/nnUNetTrainerV2__nnUNetPlansv2.1')

print(plans['plans_per_stage'])


"""

task = 'Task001_BrainTumour'
'patch_size': array([128, 128, 128])

task = 'Task006_Lung'
'patch_size': array([ 80, 192, 160])

Task007_Pancreas
'patch_size': array([ 64, 192, 192])

Task008_HepaticVessel'
array([ 64, 192, 192])

'Task010_Colon'
[ 96, 160, 160]

"""
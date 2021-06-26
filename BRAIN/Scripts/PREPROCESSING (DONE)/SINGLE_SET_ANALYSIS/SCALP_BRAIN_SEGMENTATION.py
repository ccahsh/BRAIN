
# def functions are not used purposefully. Code will be condensed once it is approved.
# pip3 is assumed to be installed. Replace pip3 with pip in if pip is used instead.
# First few lines of code will install and setup HD-BET from https://github.com/MIC-DKFZ/HD-BET and DeepBrainSeg from https://github.com/koriavinash1/DeepBrainSeg. HD-BET is the most up-to-date brain segmentation algorithm (6/16/21). DeepBrainSeg has a pretrained model for tumor segmentation that could also be run in Macbook Pro. Delete or comment on lines 21-26 once HD-BET and DeepBrainSeg are installed.
# Line 201 comment : FLIRT before brain extraction vs. FLIRT after brain extraction?
# Line 328 comment: Tumor segmentation has not been done yet due to poor processing speed in Macbook Pro. Confirming the location of saved files is necessary.


import os
import numpy as np
import nibabel as nib
from nipype.interfaces import fsl
from nipype.testing import example_data



# Task 0 : Reminding the requirements and installing HD-BET / DeepBrainSeg.

print("\nAll MNI files must be in the same directory as the python script without any additional subfolders. T1w and MNI152 reference files must be included, whereas T2w file is recommended.\n\n")

os.system("git clone https://github.com/MIC-DKFZ/HD-BET")
os.system("cd HD-BET")
os.system("pip3 install -e .")
os.system("cd ..")

os.system("pip3 install DeepBrainSeg") #put a sharp if this doesn't work in hospital intranet

from DeepBrainSeg import deepSeg #put a sharp if this doesn't work in hospital intranet

# Task 1 : Query. T1w and MNI152 files are mandatory, while T2w files are optional.

T1w_name = input("\n(REQUIRED) Type in the name of the input T1w file. Make sure you include nii.gz format.\n\n")
T2w_name = input("\n(OPTIONAL) Type in the name of the input T2w file. Make sure you include nii.gz format. Write down N/A if T2w file is not applicable or available.\n\n")
MNI_name = input("\n(REQUIRED) Type in the name of the MNI152 reference file. Make sure you include nii.gz format.\n\n")

print("\nInput complete.\n")

os.rename(MNI_name, "MNI-template.nii.gz")
os.rename(T1w_name, "input-t1w.nii.gz")

if(T2w_name == 'N/A'):
    MNIreplace = nib.load('MNI-template.nii.gz')
    MNIreplace_array = np.array(MNIreplace.dataobj)
    replace = np.zeros((MNIreplace_array.shape[0], MNIreplace_array.shape[1], MNIreplace_array.shape[2]))
    replace_nib = nib.Nifti1Image(replace, affine=np.eye(4))
    nib.save(replace_nib, "input-t2w.nii.gz")
else:
    os.rename(T2w_name, "input-t2w.nii.gz")
    
# Completion 1 notified.

confirmation1 = input("\nFile labels standardized for alignment. Press enter to continue.")
        
        
        
# Task 2 : Alignment of T1w and T2w files to MNI152 file.

flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
flt.inputs.in_file = 'input-t1w.nii.gz'
flt.inputs.reference = 'MNI-template.nii.gz'
flt.inputs.output_type = "NIFTI_GZ"
flt.cmdline
res = flt.run()

os.remove("input-t1w_flirt.mat")

flt2 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
flt2.inputs.in_file = 'input-t2w.nii.gz'
flt2.inputs.reference = 'MNI-template.nii.gz'
flt2.inputs.output_type = "NIFTI_GZ"
flt2.cmdline
res = flt2.run()

os.remove("input-t2w_flirt.mat")

flt3 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
flt3.inputs.in_file = 'input-t2w_flirt.nii.gz'
flt3.inputs.reference = 'input-t1w_flirt.nii.gz'
flt3.inputs.output_type = "NIFTI_GZ"
flt3.cmdline
res = flt3.run()

os.remove("input-t2w_flirt_flirt.mat")
os.rename("input-t2w_flirt_flirt.nii.gz", "input-t2w_flirt_t1w.nii.gz")

os.remove("input-t2w_flirt.nii.gz")

# Completion 2 notified.

confirmation2 = input("Files registered for overlay. Press enter to continue.")



# Task 3 : Overlay of MNI-aligned T1 and T2

processedt1w = nib.load('input-t1w_flirt.nii.gz')
processedt2w = nib.load('input-t2w_flirt_t1w.nii.gz')

processedt1w_array = np.array(processedt1w.dataobj)
processedt2w_array = np.array(processedt2w.dataobj)

if (processedt1w_array.shape == processedt2w_array.shape):
    confirmation31 = input("\nOverlay possible. Press enter to continue.\n")
    t1w_t2w_array = np.add(processedt1w_array, processedt2w_array)
    t1w_t2w_overlay = nib.Nifti1Image(t1w_t2w_array, affine=np.eye(4))
    nib.save(t1w_t2w_overlay, "t1w_t2w_overlay_MNIenabled.nii.gz")
    
else:
    print("Overlay not possible. Check dimensions.")

# Completion 3 notified.

confirmation3 = input("Files ready for brain mask segmentation. Press enter to continue.")



# Task 4 : Brain Mask Segmentation with HD-BET.

os.system("hd-bet -i t1w_t2w_overlay_MNIenabled.nii.gz -device cpu -mode fast -tta 0")

#remove "-device cpu -mode fast -tta 0" if GPU support is available.

os.remove("t1w_t2w_overlay_MNIenabled_bet.nii.gz")
os.rename("t1w_t2w_overlay_MNIenabled_bet_mask.nii.gz", "t1w_t2w_overlay_MNIenabled_BRAINMASK.nii.gz")

# Completion 4 notified. MNI-normalized BRAIN MASK EXTRACTION COMPLETE.

confirmation4 = input("MNI-normalized BRAIN MASK EXTRACTION COMPLETE. Saved as t1w_t2w_overlay_MNIenabled_BRAINMASK.nii.gz. Files ready for scalp segmentation. Press enter to continue.")



# Task 5 : Manual Scalp Segmentation. Parameters may be changed as necessary.

brain_mask = nib.load('t1w_t2w_overlay_MNIenabled_BRAINMASK.nii.gz')
t1w_t2w = nib.load('t1w_t2w_overlay_MNIenabled.nii.gz')

brain_mask_A = np.array(brain_mask.dataobj)
t1w_t2w_A = np.array(t1w_t2w.dataobj)

# 5.1 : Checking dimensional congruency between brain mask and overlaid file.

if(brain_mask_A.shape == t1w_t2w_A.shape):

# 5.2 : Removing brain from overlaid file.
    
    for x in range(0, brain_mask_A.shape[0]-1):
        for y in range(0, brain_mask_A.shape[1]-1):
            for z in range(0, brain_mask_A.shape[2]-1):
                if(brain_mask_A[x][y][z] > 0):
                    t1w_t2w_A[x][y][z] = 0
            
else:

    print("Comparison not possible due to difference in dimensions.")
    
# 5.3 : Isolating scalp with enclosed coordinate volume.

for x in range(0, t1w_t2w_A.shape[0]-1):
    for y in range(0, t1w_t2w_A.shape[1]-1):
        for z in range(0, t1w_t2w_A.shape[2]-1):
            if(x < ((t1w_t2w_A.shape[0]-1)*0.03) or x > ((t1w_t2w_A.shape[0]-1)*0.96) or y < ((t1w_t2w_A.shape[1]-1)*0.01) or y > ((t1w_t2w_A.shape[1]-1)*0.99) or z < ((-(t1w_t2w_A.shape[2]-1)*y*0.000275)+85)):
                t1w_t2w_A[x][y][z] = 0
                
# 5.4 : Finding value of threshold intensity for scalp segmentation.

def paraMAX():
    M = 0
    for x in range(int(0.05*(t1w_t2w_A.shape[0]-1)),int(0.95*(t1w_t2w_A.shape[0]-1))):
        for y in range(int(0.05*(t1w_t2w_A.shape[1]-1)),int(0.95*(t1w_t2w_A.shape[1]-1))):
            for z in range(int(0.05*(t1w_t2w_A.shape[2]-1)),int(0.95*(t1w_t2w_A.shape[2]-1))):
               if(M < t1w_t2w_A[x][y][z]):
                    M = t1w_t2w_A[x][y][z]
    return M
    
MAX = paraMAX()
MAX_thres = MAX*0.225

#Multiplication constant was determined from optimizing scalp segmentation from most updated pre-operative and post-operative T1w files (Patient 3) in OPENNEURO Database: https://openneuro.org/datasets/ds001226/versions/1.0.0 and https://openneuro.org/datasets/ds002080/versions/1.0.1. Patient 3 has right-sided (lateral view), parietal meningioma I. Full patient information available at Supplementary Table 1. Patient characteristics. from https://onlinelibrary.wiley.com/doi/abs/10.1002/pon.5195. For scalp segmentation, T2w files may be recommended to not be included in the input for reducing intensity values of scalp regions.

# 5.5 : Segmenting scalp using threshold intensity.

for x in range(0, t1w_t2w_A.shape[0]-1):
    for y in range(0, t1w_t2w_A.shape[1]-1):
        for z in range(0, t1w_t2w_A.shape[2]-1):
            if(t1w_t2w_A[x][y][z] < MAX_thres):
                t1w_t2w_A[x][y][z] = 0
 
scalp_array = nib.Nifti1Image(t1w_t2w_A, affine=np.eye(4))
nib.save(scalp_array, "t1w_t2w_overlay_MNIenabled_SCALP.nii.gz")

# Completion 5 notified. MNI-normalized SCALP EXTRACTION COMPLETE.

confirmation5 = input("MNI-normalized SCALP EXTRACTION COMPLETE. Saved as t1w_t2w_overlay_MNIenabled_SCALP.nii.gz. Files ready for tumor segmentation. For this segmentation, {T1w, T2w, T1ce, FLAIR} files are required. Press enter to continue.")



# Task 6 : Brain Tumor Segmentation with DeepBrainSeg. Pretrained weights will be used.
# Comment : nnuNet is ideal for brain tumor segmentation, but a simpler package is used instead so as to be implementable in Macbook Pro. Code may be changed if Hospital Intranet makes nnUNet installation possible.

# Task 6.1 : Brain extraction from MNI-aligned T1w, T2w, T1ce, and FLAIR files.
# Comment : FLIRT before brain extraction vs. FLIRT after brain extraction?

# T1w
os.system("hd-bet -i input-t1w_flirt.nii.gz -device cpu -mode fast -tta 0")
#remove "-device cpu -mode fast -tta 0" if GPU support is available.
os.rename("input-t1w_flirt_bet.nii.gz", "input-t1w_flirt_BRAIN.nii.gz")
os.remove("input-t1w_flirt_bet_mask.nii.gz")

T1w_pretumor_name = "input-t1w_flirt_BRAIN.nii.gz"

# T2w
if(T2w_name == 'N/A'):

    T2w_name = input("\n(REQUIRED) Type in the name of the input T2w file. Make sure you include nii.gz format. You cannot write N/A for this one. If you wrote N/A previously, then avoid writing input-t2w.nii.gz for this one.\n\n")
    
    flt1 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt1.inputs.in_file = T2w_name
    flt1.inputs.reference = 'MNI-template.nii.gz'
    flt1.inputs.output_type = "NIFTI_GZ"
    flt1.cmdline
    res = flt1.run()
    
    os.remove(str(T2w_name[:-7]) + "_flirt.mat")
    
    T2w_name = str(T2w_name[:-7]) + "_flirt.nii.gz"
    
    flt2 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt2.inputs.in_file = T2w_name
    flt2.inputs.reference = 'input-t1w_flirt.nii.gz'
    flt2.inputs.output_type = "NIFTI_GZ"
    flt2.cmdline
    res = flt2.run()

    os.remove(str(T2w_name[:-7]) + "_flirt.mat")
    os.rename(str(T2w_name[:-7]) + "_flirt.nii.gz", str(T2w_name[:-7]) + "_t1w.nii.gz")
    
    T2w_name = str(T2w_name[:-7]) + "_t1w.nii.gz"
    
else:

    T2w_name = "input-t2w_flirt_t1w.nii.gz"

os.system("hd-bet -i " + str(T2w_name) + " -device cpu -mode fast -tta 0")
#remove "-device cpu -mode fast -tta 0" if GPU support is available.
os.rename(str(T2w_name[:-7]) + "_bet.nii.gz", str(T2w_name[:-7]) + "_BRAIN.nii.gz")
os.remove(str(T2w_name[:-7]) + "_bet_mask.nii.gz")

T2w_pretumor_name = str(T2w_name[:-7]) + "_BRAIN.nii.gz"

#T1ce

T1ce_name = input("\n(REQUIRED) Type in the name of the input T1ce file. Make sure you include nii.gz format.\n\n")

flt1 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
flt1.inputs.in_file = T1ce_name
flt1.inputs.reference = 'MNI-template.nii.gz'
flt1.inputs.output_type = "NIFTI_GZ"
flt1.cmdline
res = flt1.run()

os.remove(str(T1ce_name[:-7]) + "_flirt.mat")

T1ce_name = str(T1ce_name[:-7]) + "_flirt.nii.gz"

flt2 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
flt2.inputs.in_file = T1ce_name
flt2.inputs.reference = 'input-t1w_flirt.nii.gz'
flt2.inputs.output_type = "NIFTI_GZ"
flt2.cmdline
res = flt2.run()

os.remove(str(T1ce_name[:-7]) + "_flirt.mat")
os.rename(str(T1ce_name[:-7]) + "_flirt.nii.gz", str(T1ce_name[:-7]) + "_t1w.nii.gz")

T1ce_name = str(T1ce_name[:-7]) + "_t1w.nii.gz"

os.system("hd-bet -i " + str(T1ce_name) + " -device cpu -mode fast -tta 0")
#remove "-device cpu -mode fast -tta 0" if GPU support is available.
os.rename(str(T1ce_name[:-7]) + "_bet.nii.gz", str(T1ce_name[:-7]) + "_BRAIN.nii.gz")
os.remove(str(T1ce_name[:-7]) + "_bet_mask.nii.gz")

T1ce_pretumor_name = str(T1ce_name[:-7]) + "_BRAIN.nii.gz"
 
#FLAIR

FLAIR_name = input("\n(REQUIRED) Type in the name of the input FLAIR file. Make sure you include nii.gz format.\n\n")

flt1 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
flt1.inputs.in_file = FLAIR_name
flt1.inputs.reference = 'MNI-template.nii.gz'
flt1.inputs.output_type = "NIFTI_GZ"
flt1.cmdline
res = flt1.run()

os.remove(str(FLAIR_name[:-7]) + "_flirt.mat")

FLAIR_name = str(FLAIR_name[:-7]) + "_flirt.nii.gz"

flt2 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
flt2.inputs.in_file = FLAIR_name
flt2.inputs.reference = 'input-t1w_flirt.nii.gz'
flt2.inputs.output_type = "NIFTI_GZ"
flt2.cmdline
res = flt2.run()

os.remove(str(FLAIR_name[:-7]) + "_flirt.mat")
os.rename(str(FLAIR_name[:-7]) + "_flirt.nii.gz", str(FLAIR_name[:-7]) + "_t1w.nii.gz")

FLAIR_name = str(FLAIR_name[:-7]) + "_t1w.nii.gz"

os.system("hd-bet -i " + str(FLAIR_name) + " -device cpu -mode fast -tta 0")
#remove "-device cpu -mode fast -tta 0" if GPU support is available.
os.rename(str(FLAIR_name[:-7]) + "_bet.nii.gz", str(FLAIR_name[:-7]) + "_BRAIN.nii.gz")
os.remove(str(FLAIR_name[:-7]) + "_bet_mask.nii.gz")

FLAIR_pretumor_name = str(FLAIR_name[:-7]) + "_BRAIN.nii.gz"
 
# Location Specification

t1_path = T1w_pretumor_name
t2_path = T2w_pretumor_name
t1ce_path = T1ce_pretumor_name
flair_path = FLAIR_pretumor_name

segmentor = deepSeg(quick=True) #put a sharp if this doesn't work in hospital intranet
segmentor.get_segmentation(t1_path, t2_path, t1ce_path, flair_path, save = True) #put a sharp if this doesn't work in hospital intranet

#Tumor segmentation has not been done yet due to poor processing speed in Macbook Pro. Confirming the location of saved files is necessary.
#Unlike HD-BET, nnUNet requires GPU, so the code cannot be done in regular laptop. Link to nnUNet: https://github.com/MIC-DKFZ/nnUNet. Installation and setup cannot be done in Macbook Pro as of right now.
#Other examples: https://github.com/Mr-TalhaIlyas/Brain-Tumor-Segmentation, https://github.com/galprz/brain-tumor-segmentation.

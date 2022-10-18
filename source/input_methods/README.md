This directory provides the code to run algorithms based on "Edge-based color constancy" by van de Weijer et al.
We provide:
* The original code by van de Weijer et al. (general_cc.m and annexed files)
* An efficient adaptation of the code, as used in our paper (general_cc_all.m and general_cc_1_1.m)
* A script to run the code for inference on the ColorChecker dataset (launch_inference.m)
* A script to quantify the inference time of the different implementations (launch_speedtest.m)

The code assumes that images in the dataset are 16-bit-encoded TIFF-RAW images, with masked-out color target.
Black level subtraction is taken care of by run_eb.m and run_eb_fast.m.
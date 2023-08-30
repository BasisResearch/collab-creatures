GENERAL INFORMATION: General overview
-------------------------------------

1. Title of dataset:
	Information integration for decision-making in desert locusts

2. Author information:
	2.1.	Yannick Günzel
			University of Konstanz
			yannick.guenzel@uni-konstanz.de
			ORCID: 0000-0001-7553-4742
		
	2.2.	Felix B. Oberhauser
			University of Konstanz
			felix.oberhauser@outlook.com
			ORCID: 0000-0002-9278-2497
	
	2.3. 	Einat Couzin-Fuchs
			University of Konstanz
			einat.couzin@uni-konstanz.de
			ORCID: 0000-0001-5269-345X

3. Date of data collection:
	2019-2020

4. Geographic location of data collection:
	Konstanz, Germany

5. Funding sources that supported the collection of the data:
	This work was completed with the support of the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy – EXC 2117 – 422037984.

6. Data and code availability
	Analysis code and processed trajectories are available at https://doi.org/10.5281/zenodo.7541780. Custom-written GUI for manually supervised tracking is available at https://github.com/YannickGuenzel/BlobMaster3000




DATA & FILE OVERVIEW
--------------------

1. Description of the dataset:
	We generated these data to investigate how desert locusts (Schistocerca gregaria) - either alone or in groups of varying sizes - accumulate evidence and integrate different classes of information for effective foraging.
	Primary video data have not been included here as this would exceed space limitations. Instead, we provide normalized 2-D trajectory data for each trial (for details, see data-specific information below).

2. Description of data analysis:
	Data were analyzed and plotted using custom-written MATLAB (R2022a) scripts (a detailed description of each script's behavior can be found below). Files that contain several sub-functions for the scripts listed above and which must be within the same folder are SubFcn.m and SubFcn_LocustDecisionSystem.m.
	To reproduce our results, run the scripts in the following order:
	1. mainFcn_LocustFeeding.m
	2. mainFcn_LocustDecisionSystem.m
	3. mainFcn_LocustDecisionSystem_plot.m		
	4. mainFcn_LocustFeeding_statistics.m		
	5. mainFcn_ExampleVideos.m		
	



METHODOLOGICAL INFORMATION & SCOPE 
----------------------------------
Locust swarms can extend over several hundred kilometers, and starvation compels this ancient pest to devour everything in its path. Theory suggests that gregarious behavior benefits foraging efficiency, yet the role of social cohesion in locust foraging decisions remains elusive. To this end, we collected high-resolution tracking data of individual and grouped gregarious desert locusts in a 2-choice behavioral assay with animals deciding between patches of either similar or different quality. Carefully maintaining the animals' identities allowed us to monitor what each individual has experienced and to estimate the leaky accumulation process of personally acquired and, when available, socially derived evidence. We fitted these data to a model based on Bayesian estimation to gain insight into the locust social decision-making system for patch selection. By disentangling the relative contribution of each information class, our study suggests that locusts balance incongruent evidence but reinforce congruent ones. We provide insight into the collective foraging decisions of social (but non-eusocial) insects and present locusts as a powerful empirical system to study individual choices and their consequent collective dynamics.




FILE-SPECIFIC INFORMATION:
--------------------------
	1. Data.zip (ZIP archive)
	   This archive contains all the (secondary) data that are needed to reproduce our results. Please consult README_Data.txt (located withing the archieve) for further details.
	   ZIP archives are open format files and can be unpacked using free and open-source file archivers
	   
	2. Analysis.zip (ZIP archive)
	   This archive contains all the Matlab (R2022a) scripts that are needed to reproduce our results. Please consult README_Analysis.txt (located withing the archieve) for further details.
	   ZIP archives are open format files and can be unpacked using free and open-source file archivers

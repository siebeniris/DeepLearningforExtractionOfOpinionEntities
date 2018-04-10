LMU Muenchen
Informatik plus Computerlinguistik
Bachelor thesis:  Deep Learning for Extraction of Opinion Entities
Author: Yiyi Chen
Supervisors:Prof. Benjamin Roth, Philipp Dufter
Examiner:Prof. Benjamin Roth
Time: 07.03.2018
==================================================================================

This CD contains the following contents:
/data - extracted datasets : train dataset, validation dataset, testdataset, train_weight (sample weight for CNN)
/data_  - extracted DSE data from  MPQA 2.0
/data_preprocessing -1) extracting data from MPQA 2.0 : search_data.py
		     2) experiment on search_data : search_holder.py
		     3) split data into 3 parts: dse_data.py
		     4) build word embedding matrix for Embedding layer: word2vector.py
		     5) transform extracted dse data into 2D arrays. 3D arrays. for model training : load_data.py

/experiments -include all the results of the experiments on models described in the thesis.

eval_text.py - present the results from models in txt files.
evaluation.py - evaluate the results from models using Binary Overlap and Proportional Overlap.

other files: models.

==============================================================================================
required libraries:
more-itertools==4.1.0 (https://github.com/erikrose/more-itertools)
keras-contrib (https://github.com/keras-team/keras-contrib)
keras==2.1.2   
tensorflow
h5py
scikit-learn
nltk
spacy
numpy
matplotlib
pytest

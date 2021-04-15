# Readme

1. Put "2017-05-12_batchdata_updated_struct_errorcorrect.mat" , "2017-06-30_batchdata_updated_struct_errorcorrect.mat","2018-04-12_batchdata_updated_struct_errorcorrect.mat" in the "Data" folder
2. Run BuildPKl_1,2,3 to derive Batch1V.pkl,Batch2V.pkl,Batch3V.pkl
3. Move the derived file to "Data" folder
4. Run FeatureExtraction.py to derive original.csv/regularized.csv
5. Run xgboostTrain.py to predict

### Requirement: 

numpy，pandas，pkl，scipy，json，h5py，matplotlib，sklearn，xgboost

### Reference:

Lifespan prediction of lithium-ion batteries based on various extracted  features and gradient boosting regression tree model




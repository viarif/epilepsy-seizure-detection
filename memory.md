本项目旨在构建一个基于颅内eeg信号硬件友好的跨患者癫痫检测模型。输入为四通道颅内EEG，一秒切窗，overlap50%；总训练参数为2,991，不进行患者适配

采用CHB-MIT的24为患者为训练，测试材料。通过line length选取最富有信息量的四条通道：F7-T7, T7-P7, F8-T8, T8-P8。通过13，14，15号患者选择固定阈值：logit `-2.632594108582`，sigmoid 约 `0.067069950728`。评估时采用LOSO策略

测试结果：Window sensitivity:50.93%  Window specificity: 97.60% (按窗口数量加权得出)
逐患者数据：


Patient Sensitivity Specificity  
 chb01  0.697708     0.992505     
 chb02  0.066667     0.994466    
 chb03  0.786477     0.985265    
 chb04  0.588624     0.986788 
 chb05  0.733634     0.983792 
 chb06  0.058333     0.961896
 chb07  0.807692     0.975047
 chb08  0.300326     0.985345 
 chb09  0.963768     0.925114
 chb10  0.848993     0.996449
 chb11  0.307398     0.980035 
 chb12  0.111064     0.987591 
 chb13  0.186905     0.978547
 chb14  0.002959     0.994693 
 chb15  0.750828     0.921335 
 chb16  0.046667     0.962462 
 chb17  0.315700     0.948031 
 chb18  0.566390     0.972173 
 chb19  0.731013     0.993034 
 chb20  0.536957     0.970259 
 chb21  0.311558     0.994912 
 chb22  0.752451     0.994100 
 chb23  0.581683     0.985312 
 chb24  0.616092     0.988853 

结果文件：

results/final_report.md
results/experiments/loso_original/loso_report.md
results/experiments/loso_original/loso_summary.json
results/experiments/loso_original/loso_per_patient.csv

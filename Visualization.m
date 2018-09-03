clear; close all; clc
EEG_MAT_DIR = 'yourlocation\imag_fbarrow_pcovmeanVPeal';
load(EEG_MAT_DIR);
clear ch*
[cnt,mrk,mnt]=eegfile_loadMatlab(EEG_MAT_DIR);
     
SMT = cntToEpo(cnt,mrk,[0000 4000]);
SMT.x=[];
SMT.y(:,1:100)=[]; 

marker={'1','left';'2','right';};
SMT.class=marker;
SMT.chan=SMT.clab;

load('yourlocation\LRP_result');

SMT.x(1,:)=relevance_test(:,5,5);   %Fp1
SMT.x(2,:)=relevance_test(:,21,5);  %Fp2
SMT.x(3,:)=relevance_test(:,9,6);    %AF3
SMT.x(4,:)=relevance_test(:,17,6);   %AF4
SMT.x(5,:)=relevance_test(:,3,8);    %F9
SMT.x(6,:)=relevance_test(:,9,8);    %F3
SMT.x(7,:)=relevance_test(:,13,8);   %Fz
SMT.x(8,:)=relevance_test(:,17,8);   %F4
SMT.x(9,:)=relevance_test(:,23,8);  %F10
SMT.x(10,:)=relevance_test(:,7,10);   %FC5
SMT.x(11,:)=relevance_test(:,9,10);   %FC3
SMT.x(12,:)=relevance_test(:,11,10);   %FC1
SMT.x(13,:)=relevance_test(:,13,10);   %FCz
SMT.x(14,:)=relevance_test(:,15,10);   %FC2
SMT.x(15,:)=relevance_test(:,17,10);   %FC4
SMT.x(16,:)=relevance_test(:,19,10);   %FC6
SMT.x(17,:)=relevance_test(:,5,12);   %T7
SMT.x(18,:)=relevance_test(:,7,12);   %C5
SMT.x(19,:)=relevance_test(:,9,12);   %C3
SMT.x(20,:)=relevance_test(:,11,12);   %C1
SMT.x(21,:)=relevance_test(:,13,12);   %Cz
SMT.x(22,:)=relevance_test(:,15,12);   %C2
SMT.x(23,:)=relevance_test(:,17,12);   %C4
SMT.x(24,:)=relevance_test(:,19,12);   %C6
SMT.x(25,:)=relevance_test(:,21,12);  %T8
SMT.x(26,:)=relevance_test(:,7,14);   %CP5
SMT.x(27,:)=relevance_test(:,9,14);   %CP3
SMT.x(28,:)=relevance_test(:,11,14);   %CP1
SMT.x(29,:)=relevance_test(:,13,14);   %CPz
SMT.x(30,:)=relevance_test(:,15,14);   %CP2
SMT.x(31,:)=relevance_test(:,17,14);   %CP4
SMT.x(32,:)=relevance_test(:,19,14);   %CP6
SMT.x(33,:)=relevance_test(:,9,16);   %P3
SMT.x(34,:)=relevance_test(:,13,16);   %Pz
SMT.x(35,:)=relevance_test(:,17,16);   %P4
SMT.x(36,:)=relevance_test(:,9,18);   %PO3
SMT.x(37,:)=relevance_test(:,17,18);   %PO4
SMT.x(38,:)=relevance_test(:,13,21);   %Oz    

rel_mean_l=zeros(38,1);
rel_mean_r=zeros(38,1);

key=0;
for trial=1:100
    if relevance_truth(1,trial)==0 
            rel_mean_l(:,1)=rel_mean_l(:,1)+SMT.x(:,trial);
            key=key+1;
    end
end
rel_mean_l(:,1)=rel_mean_l(:,1)/key;

key=0;
for trial=1:100
    if relevance_truth(1,trial)==1 
            rel_mean_r(:,1)=rel_mean_r(:,1)+SMT.x(:,trial);
            key=key+1;
    end
end
rel_mean_r(:,1)=rel_mean_r(:,1)/key;


SMT.chan(:,[3,40,41])=[];
SMT.clab(:,[3,40,41])=[];

MNT = opt_getMontage(SMT);
MNT.clab=SMT.clab;

figure(1)
plot_scalp(gca,  squeeze(rel_mean_l), MNT, [-0.4 0.4], 256);

figure(2)
plot_scalp(gca,  squeeze(rel_mean_r), MNT, [-0.4 0.4], 256);
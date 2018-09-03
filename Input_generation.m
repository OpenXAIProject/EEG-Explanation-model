clear; close all; clc

sub={'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'};
EEG_MAT_DIR = 'yourlocation\imag_fbarrow_pcovmeanVPea';

for i2=1:14
    sub={'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'};
    EEG_MAT_DIR = 'yourlocation\imag_fbarrow_pcovmeanVPea';

    load([EEG_MAT_DIR sub{i2}]);
    [cnt,mrk,mnt]=eegfile_loadMatlab([EEG_MAT_DIR sub{i2}]);
    
    smt = cntToEpo(cnt,mrk,[0000 4000]);
    smt.x=smt.x([25+1:25+301],:,:);
    smt.z = zeros(301,25,25,200);
    smt.z(:,5,5,:)=smt.x(:,1,:);   %Fp1
    smt.z(:,21,5,:)=smt.x(:,2,:);  %Fp2
    smt.z(:,9,6,:)=smt.x(:,4,:);    %AF3
    smt.z(:,17,6,:)=smt.x(:,5,:);   %AF4
    smt.z(:,3,8,:)=smt.x(:,6,:);    %F9
    smt.z(:,9,8,:)=smt.x(:,7,:);    %F3
    smt.z(:,13,8,:)=smt.x(:,8,:);   %Fz
    smt.z(:,17,8,:)=smt.x(:,9,:);   %F4
    smt.z(:,23,8,:)=smt.x(:,10,:);  %F10
    smt.z(:,7,10,:)=smt.x(:,11,:);   %FC5
    smt.z(:,9,10,:)=smt.x(:,12,:);   %FC3
    smt.z(:,11,10,:)=smt.x(:,13,:);   %FC1
    smt.z(:,13,10,:)=smt.x(:,14,:);   %FCz
    smt.z(:,15,10,:)=smt.x(:,15,:);   %FC2
    smt.z(:,17,10,:)=smt.x(:,16,:);   %FC4
    smt.z(:,19,10,:)=smt.x(:,17,:);   %FC6
    smt.z(:,5,12,:)=smt.x(:,18,:);   %T7
    smt.z(:,7,12,:)=smt.x(:,19,:);   %C5
    smt.z(:,9,12,:)=smt.x(:,20,:);   %C3
    smt.z(:,11,12,:)=smt.x(:,21,:);   %C1
    smt.z(:,13,12,:)=smt.x(:,22,:);   %Cz
    smt.z(:,15,12,:)=smt.x(:,23,:);   %C2
    smt.z(:,17,12,:)=smt.x(:,24,:);   %C4
    smt.z(:,19,12,:)=smt.x(:,25,:);   %C6
    smt.z(:,21,12,:)=smt.x(:,26,:);  %T8
    smt.z(:,7,14,:)=smt.x(:,27,:);   %CP5
    smt.z(:,9,14,:)=smt.x(:,28,:);   %CP3
    smt.z(:,11,14,:)=smt.x(:,29,:);   %CP1
    smt.z(:,13,14,:)=smt.x(:,30,:);   %CPz
    smt.z(:,15,14,:)=smt.x(:,31,:);   %CP2
    smt.z(:,17,14,:)=smt.x(:,32,:);   %CP4
    smt.z(:,19,14,:)=smt.x(:,33,:);   %CP6
    smt.z(:,9,16,:)=smt.x(:,34,:);   %P3
    smt.z(:,13,16,:)=smt.x(:,35,:);   %Pz
    smt.z(:,17,16,:)=smt.x(:,36,:);   %P4
    smt.z(:,9,18,:)=smt.x(:,37,:);   %PO3
    smt.z(:,17,18,:)=smt.x(:,38,:);   %PO4
    smt.z(:,13,21,:)=smt.x(:,39,:);   %Oz

    smt.mean=zeros(1,25,25,200);
    for j=1:1
        for i=1:300
            smt.mean(j,:,:,:)=smt.mean(j,:,:,:)+smt.z(i,:,:,:);
        end
        smt.mean(j,:,:,:)=smt.mean(j,:,:,:)/300;
    end

    clear ch*
    
    %% Dividing the data into training & test set
    num_trn = 100;       
    num_tst = size(smt.mean,4)-num_trn;

    train = smt;
    test = smt;
    train.x = smt.mean(:,:,:,1:num_trn);
    train.y = smt.y(:,1:num_trn);
    test.x = smt.mean(:,:,:,num_trn+1:end);
    test.y = smt.y(:,num_trn+1:end);
    
    train_data=zeros(100,25,25);
    for m=1:100
        for n=1:1
            train_data(n+(m-1),:,:)=train.x(n,:,:,m);
        end
    end
    
    test_data=zeros(100,25,25);
    for m=1:100
        for n=1:1
            test_data(n+(m-1),:,:)=test.x(n,:,:,m);
        end
    end
    %%
    train.y = train.y';
    train_labels=zeros(100,2);
    for a=1:1
        for b=1:100
            train_labels(a+((b-1)),:) = train.y(b,:);
        end
    end

    test.y = test.y';
    test_labels=zeros(100,2);
    for a=1:1
        for b=1:100
            test_labels(a+((b-1)),:) = test.y(b,:);
        end
    end
    
    clearvars -except test_data test_labels train_data train_labels i2 
    save (['Data_',num2str(i2),'.mat'])
end


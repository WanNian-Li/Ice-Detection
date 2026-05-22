%Search for the label class of each block result, with a maximum of 10 iterations. 
% Traverse all labels, calculate the mean and standard deviation with adjacent labels, and if the mean difference is within 0.05, consider them as the same class and merge them. 
% Finally, reassign new labels to all blocks.
function [Lmerge,Nmerge]=mergelabel(L,N,im)
    iteration=10;
    Lmerge=L;
    Nmerge=N;
    tmp=label2idx(Lmerge);
    
    for j=1:Nmerge
        pxID=tmp{j};
        pxDN=im(pxID);
        if mean(pxDN)==0
            Lmerge(tmp{j})=0;
        end
    end
    Nmerge=length(unique(Lmerge));
    if Nmerge > 100
        for i=1:iteration
            idx = label2idx(Lmerge);
            id=cellfun('length',idx);
            idx(id==0)=[];
            for labelVal = 2:Nmerge
                pxID = idx{labelVal};
                pxID_1 = idx{labelVal-1};
                pxDN=im(pxID);
                pxDN_1=im(pxID_1);
                mean1=mean(pxDN);
                mean2=mean(pxDN_1);
                if abs(mean1-mean2)<0.02
                    Lmerge(idx{labelVal-1})=labelVal;
                    idx{labelVal}=[idx{labelVal};idx{labelVal-1}];
                end
            end
            Nmerge=length(unique(Lmerge));
            if N==Nmerge || Nmerge<100
                break;
            end
            N=Nmerge;
        end
        %Reassign new labels to the merged label matrix
        label=unique(Lmerge);
        idx2 = label2idx(Lmerge);
        id=cellfun('length',idx2);
        idx2(id==0)=[];
        for j=1:length(label)
            Lmerge(idx2{j})=j;
        end
    end
end
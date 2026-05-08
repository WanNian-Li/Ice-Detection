%Determine if there are multiple peaks in the histogram
function tag=ismultipeak(x)
tag=0;
n=0;
counts=hist(x,100);
for i=4:length(counts)-3
    if counts(i)>counts(i-1) && counts(i)>counts(i+1)...
            && counts(i)>counts(i-2) &&counts(i)>counts(i+2)...
            && counts(i)>counts(i-3) && counts(i)>counts(i+3)
        temp(n+1)=i;
        n=n+1;
    end
end
if n>0
%     height=counts(temp);
%     max_c=max(counts);
%     for i=1:length(height)
%         if height(i)>max_c/2
%             tag=1;
%         end
%     end
tag=1;
end

end
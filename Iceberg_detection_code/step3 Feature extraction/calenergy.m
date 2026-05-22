%Calculate the average energy
function energy=calenergy(x)
sum=0;
for i=1:length(x)
    sum=x(i)*x(i)+sum;
end
energy=sum/length(x);
end
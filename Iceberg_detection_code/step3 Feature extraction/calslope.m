%Calculate the histogram slope
function slope=calslope(x)
[mu,sigma]=normfit(x);
y1=normpdf(mu,mu,sigma);
y2=normpdf((mu+2*sigma),mu,sigma);
slope=atan((y1-y2)/(2*sigma));
end
function x=proje(x,l)
x((x-l)<-eps)=l((x-l)<-eps);     %向l bound投影。
%x((x-u)> eps)=u((x-u)> eps);     %向u bound 投影。
%%%%%%End of projeect function
end
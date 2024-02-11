%% hessian matrix
B=[2,-1;-1,4];
g=[-1,-10]';
l=[0;0];u=[2;2];
x0=[0.5;0.5];
[xc,fcval,c,F,b]=cauchy(B,g,l,u,x0);
[x,fval]=quadprog(B,g,[],[],[],[],l,u);
%%
B1=[1,-1;-1,2];g1=[-2;-6];lb=[0;0];
[x,fval]=quadprog(B,g,[],[],[],[],l,u);
%%
B2=[2,-2,0;-2,4,0;0,0,2];g2=[0.5,0,1]';l=[0.1;0.3;0.1];u=[2;2;2];
x2=[1.5;0.5;0.5];
B=B2;g=g2;x=x2;
[xc,fcval,c,F,b,bb]=cauchy(B2,g2,l,u,x2)
[x,fval]=quadprog(B2,g2,[],[],[],[],l,u);

B=B2;g=g2;x0=x2;
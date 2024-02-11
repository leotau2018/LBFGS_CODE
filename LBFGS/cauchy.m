%% compute the Cauchy point
function [zc_e, zc_lu, dc_e, dc_lu, c, F_lu, b_lu,WMc, bb_lu]=cauchy(S, Y, g_ze, g_zlu, zb_l, z0_e, z0_lu, gamma, n, start_index, M)
%功能： 求解二次规划问的柯西点（Cauchy point），二次规划问题为：
%  min f(x)=0.5*x'*B*x+c'*x
%  s.t. l_i<=x_i<=u_i
%输入：x0是初始点，B,g分别为目标二次函数的hessian矩阵和梯度向量；
%     x0=[0.5,0.5]';    B=[2,-1;-1,4];   g=[-1,-10]';
%     l=[0,0]';u=[2,2]'; 
%     f(x)=x(1)^2-x(1)*x(2)+2*x(2)^2-x(1)-10*x(2);
%输出：zc是当前迭代点的下一个柯西点，c将被用作子问题的搜索方向,
%     F为自由分量指标集，(bb为非自有分量指标集)(可删除)。
%注：一阶导数（梯度）记为gfun(x),二阶导数记为ggfun(x);
%%%%%%%%%%%%%%%%%%%%%%%%% LBFGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%在LBFGS中，需要用到Bk=1/gamma*I-Wk*Mk*Wk' .
%  其中，Wk=[Yk,1/gamma*Sk];Mk=[-Dk,Lk';Lk,1/gamma*Sk'*Sk]^-1;
%    (Lk)_ij=s_(k-m-1+i)'*y_(k-m-1+j) when i>j; (Lk）_ij=0 otherwise;
%    Dk=diag(s_(k-m)'*y_(k-m),...,s_(k-1)'*y_(k-1));
%%%%%%%%%%%%%%%%%%%%%%%%%主程序开始%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 初始化; 
%x=x0;
%%
k_lu=length(z0_lu);                %%%原始问题的上、下约束都对应着只有下约束的对偶变量
t_lu=inf(k_lu,1);                  %
indp=find(g_zlu>eps);
t_lu(indp)=(z0_lu(indp)-zb_l(indp))./g_zlu(indp);   %所有 t_lu均大于等于零。
[t_lu_sort,ind_t_lu]=sort(t_lu);
step=find(t_lu_sort>0,1);                %t_lu等于零的情况不考虑，相应的分量不移动。


dc_e=-g_ze;
dc_lu=-g_zlu;
dc_lu(t_lu<eps )=0;
dc=[dc_e;dc_lu];
%F_lu=find(t_lu>0);
 


if n==0                                                          %
    return;
else
    if n==M && start_index>1
        ind=[(start_index:M), 1:start_index-1];
        S=S(:,ind);
        Y=Y(:,ind);
    else
        ind=1:n;
        S=S(:,ind);
        Y=Y(:,ind);
    end
end
Wk=[Y,S/gamma];                                                      %LBFGS相关变量计算

SY=S'*Y;
Dk=diag(diag(SY));
Lk=tril(SY,-1);
Mk=speye(2*size(SY))/([-Dk,Lk';Lk,1/gamma*(S'*S)]);%与后者比速度更快吗？ Mk=inv([-Dk,Lk';Lk,1/gamma*(S'*S)])
p=Wk'*dc;
c=0;                                                    %其实c和P的维数相同，都是M*1的向量。
gfun=-dc'*dc;                                             %%%gfun于ggfun 的计算是否依然考虑等式约束部分？
ggfun=-1/gamma*gfun -p'*Mk*p;     %LBFGS方法
dt_min =-gfun/ggfun;
t_old_lu=0;

tt_lu=t_lu_sort(step);
%[tt_lu,b_lu]=min(t_lu(F_lu));
%tb_lu=b_lu;
b_lu=ind_t_lu(step);           %b_lu为对应应该剔出的分量。
bb_lu=b_lu;                     %第一个被剔出的分量
%F_lu(ind_t_lu(step))=[];     %remove b from F if ti=t;
dt_lu=tt_lu;            %第一小步，tt_lu不为零

zc_e=z0_e;zc_lu=z0_lu;
zz=zeros(k_lu,1);
%% Examination of subsequent segments
while dt_min>=dt_lu
    if dc_lu(b_lu)<0,   zc_lu(b_lu)=zb_l(b_lu);   end
    zz(b_lu)=zc_lu(b_lu)-z0_lu(b_lu);
    c=c+dt_lu*p;
    %gfun=gfun+dt*ggfun+g(b)^2+g(b)*B(b,:)*z;         %更新一阶导gfun
    %ggfun=ggfun+2*g(b)*B(b,:)*d+g(b)^2*B(b,b);          %更新二阶导gfun
    gfun=gfun+dt_lu*ggfun+g_zlu(b_lu)^2+1/gamma*g_zlu(b_lu)*zz(b_lu)-g_zlu(b_lu)*Wk(b_lu,:)*Mk*c;        %LBFGSf更新一阶导
    ggfun=ggfun-1/gamma * g_zlu(b_lu)^2 - 2 * g_zlu(b_lu) * Wk(b_lu,:)* Mk* p-g_zlu(b_lu)^2 * Wk(b_lu,:)*Mk*Wk(b_lu,:)';   %LBFGSf更新二阶导
    p=p+g_zlu(b_lu)*Wk(b_lu,:)';                                                   %%LBFGS更新p
    dc_lu(b_lu)=0;
    dt_min=-gfun/ggfun;
    t_old_lu=tt_lu;
    step=step+1;
    tt_lu=t_lu_sort(step);
    b_lu=ind_t_lu(step);                     %%此处针对临近的t_lu_sort想等或距离很近时，也可以应对。但相当于重复循环。
    %[tt_lu,b_lu]=min(t_lu(F_lu));
    %tb_lu=b_lu;
    %b_lu=F_lu(b_lu);
    %bb_lu=[bb_lu;b_lu];
    %F_lu(tb_lu)=[];     %remove b from F if ti=t;
    dt_lu=tt_lu-t_old_lu;
end
%%
%step=step+norm(abs(t_lu_sort-t_lu_sort(step))<eps);
F_lu=ind_t_lu(step:end);
bb_lu=ind_t_lu(1:step-1);
dt_min=max(dt_min,0);
t_old_lu=t_old_lu+dt_min;
zc_lu(F_lu) =zc_lu(F_lu)  + t_old_lu*dc_lu(F_lu);
zc_e        =zc_e         + t_old_lu*dc_e;

%F_lu(abs(t_lu(F_lu)-tt_lu)<eps)=[];     %remove i from F if ti=t;
c=c+dt_min*p;                                                           %%LBFGS更新c

WMc=Wk*(Mk*c);                        % output WMc for gredient computeing at cauchy point
end


%%%%%%注意%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



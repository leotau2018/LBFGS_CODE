%% ����d2�ı�׼�����з����ֳ��ĸ�ָ�꼯��
%C1Ϊ��ЧԼ����,��������Ϊ�㣻C2Ϊ�ٽ��߽��ָ�꼯����������Ϊ���ݶȷ���C3Ϊ�ٽ��߽�Ļ���Լ������������Ϊ����ĵִ�߽緽��
function [CC1,CC2,B]=classify2(zc_lu, gp_zlu, zb_l, w)
em=min(w,1e-6);
CC1=find(abs(zc_lu-zb_l)<=em  & gp_zlu>-em);
CC2=find(abs(zc_lu-zb_l)<=em  & gp_zlu<=-em);
% C3=find(abs(zc_lu-zb_l)>0    & abs(zc_lu-zb_l)<=em  & gp_zlu>=0);
% C=[CC1;CC2;C3];
C=[CC1;CC2];
B =(1:1:length(zc_lu))';
B(C)=[];
end
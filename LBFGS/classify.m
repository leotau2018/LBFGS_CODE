%% ����d1�ı�׼�����з����ֳ��ĸ�ָ�꼯��
%C1Ϊ��ЧԼ����,��������Ϊ�㣻C2Ϊ�ٽ��߽��ָ�꼯����������Ϊ���ݶȷ���C3Ϊ�ٽ��߽�Ļ���Լ������������Ϊ����ĵִ�߽緽��
function [C1,C2,C3,B]=classify(zc_lu, gp_zlu, zb_l, w)
em=min(w,1e-6);
C1=find(abs(zc_lu-zb_l)<eps  & gp_zlu>-eps);
C2=find(abs(zc_lu-zb_l)<=em  & gp_zlu<0);
C3=find(abs(zc_lu-zb_l)>0    & abs(zc_lu-zb_l)<=em  & gp_zlu>=0);
C=[C1;C2;C3];
B =(1:1:length(zc_lu))';
B(C)=[];
end